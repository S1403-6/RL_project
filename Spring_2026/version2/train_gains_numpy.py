import numpy as np
import json
import os
from feature_extractor import CNNFeatureExtractor, SyntheticFrameGenerator

EPISODES  = 600
MAX_STEPS = 300
GAMMA     = 0.99
LR        = 0.001
INPUT_DIM = 16

np.random.seed(42)

class NumpyNeuralNet:
    def __init__(self, layer_sizes=[16, 128, 64, 2], lr=0.001):
        self.lr          = lr
        self.layer_sizes = layer_sizes
        self.weights     = []
        self.biases      = []

        for i in range(len(layer_sizes) - 1):
            fan_in  = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            W = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            b = np.zeros((1, fan_out))
            self.weights.append(W)
            self.biases.append(b)

        self.m_w = [np.zeros_like(w) for w in self.weights]
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]
        self.t   = 0

    def relu(self, x):        return np.maximum(0, x)
    def relu_grad(self, x):   return (x > 0).astype(float)
    def softplus(self, x):    return np.where(x > 20, x, np.log1p(np.exp(np.clip(x, -500, 20))))
    def softplus_grad(self, x): return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, x):
        x = x.reshape(1, -1)
        cache = {'inputs': [x], 'pre_activations': []}
        current = x
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = current @ W + b
            cache['pre_activations'].append(z)
            if i < len(self.weights) - 1:
                current = self.relu(z)
            else:
                current = self.softplus(z) + 0.5
            cache['inputs'].append(current)
        return current.flatten(), cache

    def backward(self, cache, advantage, Kp_noisy, Kd_noisy, Kp_mean, Kd_mean, std=0.1):
        gains_mean  = np.array([Kp_mean,  Kd_mean ]).reshape(1, 2)
        gains_noisy = np.array([Kp_noisy, Kd_noisy]).reshape(1, 2)
        d_log_pi = (gains_noisy - gains_mean) / (std ** 2)
        delta    = -d_log_pi * advantage
        z_last   = cache['pre_activations'][-1]
        delta    = delta * self.softplus_grad(z_last)
        grad_W, grad_b = [], []
        for i in reversed(range(len(self.weights))):
            inp = cache['inputs'][i]
            grad_W.insert(0, np.clip(inp.T @ delta, -1.0, 1.0))
            grad_b.insert(0, np.clip(delta.copy(),  -1.0, 1.0))
            if i > 0:
                delta = delta @ self.weights[i].T * self.relu_grad(cache['pre_activations'][i-1])
        return grad_W, grad_b

    def adam_update(self, grad_W, grad_b, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        lr_t = self.lr * np.sqrt(1 - beta2**self.t) / (1 - beta1**self.t)
        for i in range(len(self.weights)):
            self.m_w[i] = beta1*self.m_w[i] + (1-beta1)*grad_W[i]
            self.v_w[i] = beta2*self.v_w[i] + (1-beta2)*grad_W[i]**2
            self.weights[i] -= lr_t * self.m_w[i] / (np.sqrt(self.v_w[i]) + eps)
            self.m_b[i] = beta1*self.m_b[i] + (1-beta1)*grad_b[i]
            self.v_b[i] = beta2*self.v_b[i] + (1-beta2)*grad_b[i]**2
            self.biases[i] -= lr_t * self.m_b[i] / (np.sqrt(self.v_b[i]) + eps)

    def save_weights(self, path="nn_weights.npz"):
        arrays = {f"W{i}": w for i, w in enumerate(self.weights)}
        arrays.update({f"b{i}": b for i, b in enumerate(self.biases)})
        np.savez(path, **arrays)

    def load_weights(self, path="nn_weights.npz"):
        data = np.load(path)
        for i in range(len(self.weights)):
            self.weights[i] = data[f"W{i}"]
            self.biases[i]  = data[f"b{i}"]


class LightSim:
    def __init__(self):
        self.frame_gen = SyntheticFrameGenerator()
        self.extractor = CNNFeatureExtractor()
        self.reset()

    def reset(self):
        self.pos        = np.array([0.0, 0.0])
        self.vel        = np.array([0.0, 0.0])
        self.yaw        = 0.0
        self.target     = np.array([6.0, 0.0])
        self.step_count = 0
        self.extractor.prev_gray = None
        self.init_obs = np.array([
            [1.8,  1.2], [1.8, -1.2],
            [2.8,  0.6], [2.8, -0.6],
            [3.6,  1.4], [3.6, -1.4],
        ])
        self.obs_pos = self.init_obs.copy()
        return self._get_image_features()

    def _move_obstacles(self):
        for idx in range(len(self.init_obs)):
            self.obs_pos[idx, 1] = (self.init_obs[idx, 1]
                + 0.5 * np.sin(self.step_count * 0.02 + idx * 0.7))

    def _get_image_features(self):
        """Render synthetic frame → extract 16 image features."""
        frame = self.frame_gen.render(
            robot_pos     = self.pos,
            robot_yaw     = self.yaw,
            target_pos    = self.target,
            obs_positions = self.obs_pos.tolist()
        )
        return self.extractor.extract(frame)   # (16,)

    def _lyapunov_step(self, Kp, Kd):
        K_b = 2.0;  K_theta = 2.0
        pos_error    = self.target - self.pos
        dist_to_goal = np.linalg.norm(pos_error)
        if dist_to_goal < 0.1:
            return 0.0, 0.0, dist_to_goal
        u_att     = pos_error * Kp - self.vel * Kd
        dists_obs = sorted([(np.linalg.norm(self.pos - o), o)
                            for o in self.obs_pos], key=lambda x: x[0])[:3]
        u_barrier = np.zeros(2)
        for d_center, obs in dists_obs:
            obs_vec = self.pos - obs
            if d_center < 1e-3: continue
            h = d_center - 0.75
            if h < 1.0:
                h           = max(h, 0.02)
                rep_dir     = obs_vec / d_center
                barrier_mag = K_b / (h+0.5) ** 2
                t1 = np.array([-rep_dir[1], rep_dir[0]])
                t2 = np.array([ rep_dir[1],-rep_dir[0]])
                tangent     = t1 if np.dot(t1, pos_error) > np.dot(t2, pos_error) else t2
                u_barrier  += barrier_mag * rep_dir + 2.0 * barrier_mag * tangent
        u_total   = u_att + u_barrier + np.array([0.0, 0.15])
        theta_des = np.arctan2(u_total[1], u_total[0])
        e_theta   = np.arctan2(np.sin(theta_des-self.yaw), np.cos(theta_des-self.yaw))
        v         = min(np.linalg.norm(u_total), 1.2)
        omega     = np.clip(K_theta * e_theta, -2.0, 2.0)
        return v, omega, dist_to_goal

    def step(self, Kp, Kd):
        dt = 1 / 60.0
        self._move_obstacles()
        v, omega, dist = self._lyapunov_step(Kp, Kd)
        self.vel        = np.array([v*np.cos(self.yaw), v*np.sin(self.yaw)])
        self.pos        = self.pos + self.vel * dt
        self.yaw        = self.yaw + omega * dt
        self.step_count += 1
        collision = any(np.linalg.norm(self.pos - o) < 0.5 for o in self.obs_pos)
        done      = dist < 0.2 or collision or self.step_count >= MAX_STEPS
        return self._get_image_features(), dist, collision, done


def compute_reward(dist, prev_dist, collision, speed, reached_goal):
    r  = (prev_dist - dist) * 10.0
    r -= 0.02 * speed**2
    if collision:    r -= 50.0
    if reached_goal: r += 100.0
    return r

def compute_returns(rewards, gamma=0.99):
    G, returns = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = np.array(returns)
    if returns.std() > 1e-8:
        returns = (returns - returns.mean()) / returns.std()
    return returns


def train():
    net = NumpyNeuralNet(layer_sizes=[INPUT_DIM, 128, 64, 2], lr=LR)
    sim = LightSim()
    episode_rewards = []

    print("=" * 60)
    print("  Training  |  Camera Frame → Features (16) → NN → [Kp,Kd]")
    print("=" * 60)

    for ep in range(EPISODES):
        obs       = sim.reset()
        prev_dist = np.linalg.norm(sim.target - sim.pos)
        caches, kp_means, kd_means, kp_noisys, kd_noisys, rewards_ep = [], [], [], [], [], []

        for _ in range(MAX_STEPS):
            gains, cache = net.forward(obs)       #
            Kp_mean, Kd_mean = float(gains[0]), float(gains[1])

            std      = max(0.3 * (1 - ep / EPISODES), 0.05)
            Kp_noisy = np.clip(Kp_mean + np.random.randn()*std, 0.5, 8.0)
            Kd_noisy = np.clip(Kd_mean + np.random.randn()*std, 0.5, 5.0)

            next_obs, dist, collision, done = sim.step(Kp_noisy, Kd_noisy)
            reward = compute_reward(dist, prev_dist, collision,
                                    np.linalg.norm(sim.vel), dist < 0.2)

            caches.append(cache); kp_means.append(Kp_mean); kd_means.append(Kd_mean)
            kp_noisys.append(Kp_noisy); kd_noisys.append(Kd_noisy); rewards_ep.append(reward)
            prev_dist = dist;  obs = next_obs
            if done: break

        returns      = compute_returns(rewards_ep, GAMMA)
        total_grad_W = [np.zeros_like(w) for w in net.weights]
        total_grad_b = [np.zeros_like(b) for b in net.biases]

        for cache, kpm, kdm, kpn, kdn, Gt in zip(
                caches, kp_means, kd_means, kp_noisys, kd_noisys, returns):
            gW, gb = net.backward(cache, Gt, kpn, kdn, kpm, kdm, std=0.1)
            for i in range(len(net.weights)):
                total_grad_W[i] += gW[i]; total_grad_b[i] += gb[i]

        n = len(caches)
        net.adam_update([g/n for g in total_grad_W], [g/n for g in total_grad_b])

        episode_rewards.append(sum(rewards_ep))
        if (ep + 1) % 50 == 0:
            avg        = np.mean(episode_rewards[-50:])
            test_frame = sim.frame_gen.render([3.0,0.0], 0.0, [6.0,0.0],
                                              [[2.8,0.6],[2.8,-0.6],[3.6,1.4]])
            g, _       = net.forward(sim.extractor.extract(test_frame))
            print(f"  Ep {ep+1:4d}/{EPISODES} | Avg Reward: {avg:8.2f} | "
                  f"Kp={g[0]:.3f}  Kd={g[1]:.3f}")

    # --- final evaluation on rendered test scenes ---
    print("\n--- Final Gains on Test Scenes ---")
    scenes = [
        ("Near start, obstacles close",  [0.5,0.0], 0.0, [6.0,0.0], [[1.8,1.2],[1.8,-1.2]]),
        ("Mid path,   obstacles nearby", [3.0,0.0], 0.0, [6.0,0.0], [[2.8,0.6],[3.6, 1.4]]),
        ("Near goal,  open space",       [5.0,0.0], 0.0, [6.0,0.0], [[3.6,-1.4]]),
    ]
    kp_list, kd_list = [], []
    for label, rpos, ryaw, tpos, obs in scenes:
        feats = sim.extractor.extract(sim.frame_gen.render(rpos, ryaw, tpos, obs))
        g, _  = net.forward(feats)
        kp_list.append(float(g[0])); kd_list.append(float(g[1]))
        print(f"  {label:35s} → Kp={g[0]:.4f},  Kd={g[1]:.4f}")

    Kp_final, Kd_final = float(np.mean(kp_list)), float(np.mean(kd_list))
    net.save_weights("nn_weights.npz")
    with open("trained_gains.json", "w") as f:
        json.dump({"Kp": round(Kp_final,4), "Kd": round(Kd_final,4),
                   "note": "Image-based NN. Use nn_weights.npz for dynamic gains."}, f, indent=4)
    print(f"\n  Saved → nn_weights.npz  (copy this to laptop)")
    print(f"  Saved → trained_gains.json")
    print(f"  Final Kp={Kp_final:.4f},  Kd={Kd_final:.4f}")


if __name__ == "__main__":
    train()