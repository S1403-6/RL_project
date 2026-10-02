import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class AIRLDiscriminator(nn.Module):

    def __init__(
        self,
        state_dim=32,
        action_dim=2,
        hidden_dim=128,
        gamma=0.99,
        lr=3e-4,
        value_lr=None,
        value_weight_decay=1e-4,
        reward_weight_decay=1e-4,
        value_reg_weight=1e-5,
        temporal_smoothness_weight=1e-4,
        h_clip=5.0
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        self.value_reg_weight = value_reg_weight
        self.temporal_smoothness_weight = temporal_smoothness_weight
        self.h_clip = h_clip

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # g(s,a): non-potential reward component.
        self.reward_network = nn.Sequential(
            nn.Linear(
                state_dim + action_dim,
                hidden_dim
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim // 2
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim // 2,
                1
            )
        )

        # h(s): AIRL potential / state shaping function.
        #
        # This is NOT PPO's value critic. Its role is to provide:
        #
        #     gamma * h(s') - h(s)
        #
        # inside the AIRL reward.
        self.value_network = nn.Sequential(
            nn.Linear(
                state_dim,
                hidden_dim
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim // 2
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim // 2,
                1
            )
        )

        self.to(self.device)

        if value_lr is None:
            value_lr = lr

        # Separate optimizers make it possible to control the potential
        # network independently from g(s,a).
        self.reward_optimizer = optim.Adam(
            self.reward_network.parameters(),
            lr=lr,
            weight_decay=reward_weight_decay
        )

        self.value_optimizer = optim.Adam(
            self.value_network.parameters(),
            lr=value_lr,
            weight_decay=value_weight_decay
        )

        # Kept for compatibility with existing training code:
        # discriminator.loss_fn = ...
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.last_stats = {}

    def _to_tensor(self, x):

        if torch.is_tensor(x):
            tensor = x.float()

        elif isinstance(x, np.ndarray):
            tensor = torch.from_numpy(
                x.astype(np.float32)
            )

        else:
            tensor = torch.tensor(
                x,
                dtype=torch.float32
            )

        if tensor.ndim == 0:
            tensor = tensor.reshape(1, 1)

        elif tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)

        return tensor.to(self.device)

    def _to_done_tensor(self, done):

        if torch.is_tensor(done):
            tensor = done.float()

        elif isinstance(done, np.ndarray):
            tensor = torch.from_numpy(
                done.astype(np.float32)
            )

        else:
            tensor = torch.tensor(
                done,
                dtype=torch.float32
            )

        if tensor.ndim == 0:
            tensor = tensor.reshape(1, 1)

        elif tensor.ndim == 1:
            tensor = tensor.unsqueeze(-1)

        return tensor.to(self.device)

    def reward_function(
        self,
        state,
        action
    ):

        state = self._to_tensor(state)
        action = self._to_tensor(action)

        sa = torch.cat(
            [state, action],
            dim=-1
        )

        return self.reward_network(sa)

    def value_function(
        self,
        state
    ):

        state = self._to_tensor(state)

        value = self.value_network(state)

        # Bounding the potential prevents h(s) from becoming arbitrarily
        # large and dominating gamma*h(s') - h(s).
        if self.h_clip is not None:
            value = self.h_clip * torch.tanh(value / self.h_clip)

        return value

    def potential_difference(
        self,
        state,
        next_state,
        done
    ):

        state = self._to_tensor(state)
        next_state = self._to_tensor(next_state)
        done = self._to_done_tensor(done)

        h_s = self.value_function(state)
        h_next = self.value_function(next_state)

        shaping = (
            self.gamma
            * (1.0 - done)
            * h_next
            - h_s
        )

        return shaping, h_s, h_next

    def forward(
        self,
        state,
        action,
        next_state,
        done
    ):

        reward = self.reward_function(
            state,
            action
        )

        shaping, _, _ = self.potential_difference(
            state,
            next_state,
            done
        )

        return reward + shaping

    def discriminator_logits(
        self,
        state,
        action,
        next_state,
        done,
        log_policy_prob
    ):

        f = self.forward(
            state,
            action,
            next_state,
            done
        )

        log_policy_prob = self._to_tensor(
            log_policy_prob
        )

        return f - log_policy_prob

    def discriminator_probability(
        self,
        state,
        action,
        next_state,
        done,
        log_policy_prob
    ):

        self.eval()

        with torch.no_grad():

            logits = self.discriminator_logits(
                state,
                action,
                next_state,
                done,
                log_policy_prob
            )

            probability = torch.sigmoid(
                logits
            )

        probability = (
            probability
            .cpu()
            .numpy()
            .squeeze()
        )

        if np.ndim(probability) == 0:
            return float(probability)

        return probability

    def train_step(
        self,
        expert_states,
        expert_actions,
        expert_next_states,
        expert_dones,
        expert_log_probs,
        agent_states,
        agent_actions,
        agent_next_states,
        agent_dones,
        agent_log_probs
    ):

        self.train()

        # ---------------------------------------------------------------
        # Convert everything once.
        # ---------------------------------------------------------------
        expert_states = self._to_tensor(expert_states)
        expert_actions = self._to_tensor(expert_actions)
        expert_next_states = self._to_tensor(expert_next_states)
        expert_dones = self._to_done_tensor(expert_dones)
        expert_log_probs = self._to_tensor(expert_log_probs)

        agent_states = self._to_tensor(agent_states)
        agent_actions = self._to_tensor(agent_actions)
        agent_next_states = self._to_tensor(agent_next_states)
        agent_dones = self._to_done_tensor(agent_dones)
        agent_log_probs = self._to_tensor(agent_log_probs)

        # ---------------------------------------------------------------
        # AIRL discriminator objective.
        # ---------------------------------------------------------------
        expert_logits = self.discriminator_logits(
            expert_states,
            expert_actions,
            expert_next_states,
            expert_dones,
            expert_log_probs
        )

        agent_logits = self.discriminator_logits(
            agent_states,
            agent_actions,
            agent_next_states,
            agent_dones,
            agent_log_probs
        )

        expert_targets = torch.ones_like(
            expert_logits,
            device=self.device
        )

        agent_targets = torch.zeros_like(
            agent_logits,
            device=self.device
        )

        expert_loss = self.loss_fn(
            expert_logits,
            expert_targets
        )

        agent_loss = self.loss_fn(
            agent_logits,
            agent_targets
        )

        discriminator_loss = (
            expert_loss
            + agent_loss
        )

        # ---------------------------------------------------------------
        # Explicit regularization for h(s).
        #
        # 1. Magnitude regularization prevents the potential from growing
        #    without bound.
        #
        # 2. Temporal smoothness discourages abrupt changes in h(s) between
        #    consecutive states while still allowing useful shaping.
        #
        # These coefficients are deliberately small so that AIRL's
        # discriminator objective remains dominant.
        # ---------------------------------------------------------------
        expert_h = self.value_function(
            expert_states
        )

        expert_h_next = self.value_function(
            expert_next_states
        )

        agent_h = self.value_function(
            agent_states
        )

        agent_h_next = self.value_function(
            agent_next_states
        )

        all_h = torch.cat(
            [expert_h, agent_h],
            dim=0
        )

        value_magnitude_loss = torch.mean(
            all_h.pow(2)
        )

        expert_temporal_delta = (
            expert_h_next
            - expert_h
        )

        agent_temporal_delta = (
            agent_h_next
            - agent_h
        )

        temporal_smoothness_loss = torch.mean(
            torch.cat(
                [
                    expert_temporal_delta,
                    agent_temporal_delta
                ],
                dim=0
            ).pow(2)
        )

        regularization_loss = (
            self.value_reg_weight
            * value_magnitude_loss
            +
            self.temporal_smoothness_weight
            * temporal_smoothness_loss
        )

        total_loss = (
            discriminator_loss
            + regularization_loss
        )

        # ---------------------------------------------------------------
        # One joint backward pass.
        # ---------------------------------------------------------------
        self.reward_optimizer.zero_grad(set_to_none=True)
        self.value_optimizer.zero_grad(set_to_none=True)

        total_loss.backward()

        reward_grad_norm = nn.utils.clip_grad_norm_(
            self.reward_network.parameters(),
            max_norm=1.0
        )

        value_grad_norm = nn.utils.clip_grad_norm_(
            self.value_network.parameters(),
            max_norm=1.0
        )

        self.reward_optimizer.step()
        self.value_optimizer.step()

        # ---------------------------------------------------------------
        # Diagnostics.
        # ---------------------------------------------------------------
        with torch.no_grad():

            expert_h_mean = float(
                expert_h.mean().item()
            )

            agent_h_mean = float(
                agent_h.mean().item()
            )

            expert_h_std = float(
                expert_h.std(unbiased=False).item()
            )

            agent_h_std = float(
                agent_h.std(unbiased=False).item()
            )

            expert_shaping = (
                self.gamma
                * (1.0 - expert_dones)
                * expert_h_next
                - expert_h
            )

            agent_shaping = (
                self.gamma
                * (1.0 - agent_dones)
                * agent_h_next
                - agent_h
            )

            expert_shaping_mean = float(
                expert_shaping.mean().item()
            )

            agent_shaping_mean = float(
                agent_shaping.mean().item()
            )

            expert_shaping_std = float(
                expert_shaping.std(unbiased=False).item()
            )

            agent_shaping_std = float(
                agent_shaping.std(unbiased=False).item()
            )

            expert_probability = torch.sigmoid(
                expert_logits
            )

            agent_probability = torch.sigmoid(
                agent_logits
            )

            self.last_stats = {
                "total_loss": float(total_loss.item()),
                "discriminator_loss": float(discriminator_loss.item()),
                "expert_loss": float(expert_loss.item()),
                "agent_loss": float(agent_loss.item()),
                "value_magnitude_loss": float(
                    value_magnitude_loss.item()
                ),
                "temporal_smoothness_loss": float(
                    temporal_smoothness_loss.item()
                ),
                "regularization_loss": float(
                    regularization_loss.item()
                ),
                "expert_h_mean": expert_h_mean,
                "agent_h_mean": agent_h_mean,
                "expert_h_std": expert_h_std,
                "agent_h_std": agent_h_std,
                "expert_shaping_mean": expert_shaping_mean,
                "agent_shaping_mean": agent_shaping_mean,
                "expert_shaping_std": expert_shaping_std,
                "agent_shaping_std": agent_shaping_std,
                "expert_probability": float(
                    expert_probability.mean().item()
                ),
                "agent_probability": float(
                    agent_probability.mean().item()
                ),
                "reward_grad_norm": float(reward_grad_norm),
                "value_grad_norm": float(value_grad_norm)
            }

        return (
            float(total_loss.item()),
            float(expert_loss.item()),
            float(agent_loss.item())
        )

    def get_diagnostics(self):
        """Return the most recent h(s) and discriminator diagnostics."""
        return dict(self.last_stats)

    def predict_reward(
        self,
        state,
        action,
        next_state=None,
        done=None
    ):

        self.eval()

        with torch.no_grad():

            if next_state is not None and done is not None:

                reward = self.forward(
                    state,
                    action,
                    next_state,
                    done
                )

            else:

                reward = self.reward_function(
                    state,
                    action
                )

        reward = (
            reward
            .cpu()
            .numpy()
            .squeeze()
        )

        if np.ndim(reward) == 0:
            return float(reward)

        return reward

    def predict_components(
        self,
        state,
        action,
        next_state,
        done
    ):
        self.eval()

        with torch.no_grad():
            g = self.reward_function(
                state,
                action
            )

            shaping, h_s, h_next = (
                self.potential_difference(
                    state,
                    next_state,
                    done
                )
            )

            f = g + shaping

        def to_numpy(x):
            value = (
                x.cpu()
                .numpy()
                .squeeze()
            )
            if np.ndim(value) == 0:
                return float(value)
            return value

        return (
            to_numpy(g),
            to_numpy(shaping),
            to_numpy(f),
            to_numpy(h_s),
            to_numpy(h_next)
        )


    def predict_raw_reward(
        self,
        state,
        action
    ):

        self.eval()

        with torch.no_grad():

            reward = self.reward_function(
                state,
                action
            )

        reward = (
            reward
            .cpu()
            .numpy()
            .squeeze()
        )

        if np.ndim(reward) == 0:
            return float(reward)

        return reward

    def predict_potential(
        self,
        state
    ):

        self.eval()

        with torch.no_grad():

            h = self.value_function(
                state
            )

        h = (
            h
            .cpu()
            .numpy()
            .squeeze()
        )

        if np.ndim(h) == 0:
            return float(h)

        return h

    def predict_shaping(
        self,
        state,
        next_state,
        done
    ):

        self.eval()

        with torch.no_grad():

            shaping, _, _ = self.potential_difference(
                state,
                next_state,
                done
            )

        shaping = (
            shaping
            .cpu()
            .numpy()
            .squeeze()
        )

        if np.ndim(shaping) == 0:
            return float(shaping)

        return shaping
