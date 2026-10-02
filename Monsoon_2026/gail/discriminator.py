import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class GAILDiscriminator(nn.Module):

    def __init__(
        self,
        state_dim=32,
        action_dim=2,
        hidden_dim=128,
        lr=0.0003
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.network = nn.Sequential(
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
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.parameters(),
            lr=lr,
            weight_decay=1e-4
        )

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, state, action):

        if isinstance(state, np.ndarray):
            state = torch.from_numpy(
                state.astype(np.float32)
            )

        if isinstance(action, np.ndarray):
            action = torch.from_numpy(
                action.astype(np.float32)
            )

        if state.ndim == 1:
            state = state.unsqueeze(0)

        if action.ndim == 1:
            action = action.unsqueeze(0)

        state = state.float().to(self.device)
        action = action.float().to(self.device)

        sa = torch.cat(
            [state, action],
            dim=-1
        )

        return self.network(sa)

    def predict_probability(
        self,
        state,
        action
    ):

        self.eval()

        with torch.no_grad():

            logits = self.forward(
                state,
                action
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
        agent_states,
        agent_actions
    ):

        self.train()

        self.optimizer.zero_grad()

        expert_logits = self.forward(
            expert_states,
            expert_actions
        )

        agent_logits = self.forward(
            agent_states,
            agent_actions
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

        total_loss = (
            expert_loss
            + agent_loss
        )

        total_loss.backward()

        nn.utils.clip_grad_norm_(
            self.parameters(),
            max_norm=1.0
        )

        self.optimizer.step()

        return (
            float(total_loss.item()),
            float(expert_loss.item()),
            float(agent_loss.item())
        )

    def predict_reward(
        self,
        state,
        action
    ):

        self.eval()

        with torch.no_grad():

            logits = self.forward(
                state,
                action
            )

            probability = torch.sigmoid(
                logits
            )

            reward = -torch.log(
                1.0 - probability + 1e-8
            )

            reward_np = (
                reward
                .cpu()
                .numpy()
                .squeeze()
            )

        if np.ndim(reward_np) == 0:
            return float(reward_np)

        return reward_np