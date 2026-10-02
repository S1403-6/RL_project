import numpy as np
import os


class ImitationPolicyNet:
    def __init__(
        self,
        layer_sizes=[28, 128, 64, 2],
        lr=0.001,
        max_v=4.0,
        max_omega=2.5
    ):
        self.layer_sizes = layer_sizes
        self.lr = lr
        self.max_v = max_v
        self.max_omega = max_omega

        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]

            W = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            b = np.zeros((1, fan_out))

            self.weights.append(W)
            self.biases.append(b)

        self.m_w = [np.zeros_like(w) for w in self.weights]
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]

        self.t = 0

        self.mean = np.zeros(layer_sizes[0], dtype=np.float32)
        self.std = np.ones(layer_sizes[0], dtype=np.float32)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_grad(self, x):
        return (x > 0).astype(np.float32)

    def normalize(self, x):
        return (x - self.mean) / (self.std + 1e-6)

    def normalize_actions(self, actions):
        actions = np.asarray(actions, dtype=np.float32)

        v = actions[:, 0:1]
        omega = actions[:, 1:2]

        v_norm = 2.0 * (v / self.max_v) - 1.0
        omega_norm = omega / self.max_omega

        return np.hstack([v_norm, omega_norm])

    def denormalize_actions(self, actions):
        actions = np.asarray(actions, dtype=np.float32)

        v_norm = actions[:, 0:1]
        omega_norm = actions[:, 1:2]

        v = (v_norm + 1.0) * 0.5 * self.max_v
        omega = omega_norm * self.max_omega

        return np.hstack([v, omega])

    def forward(self, x, is_normalized=False):
        if x.ndim == 1:
            x = x.reshape(1, -1)

        if not is_normalized:
            x = self.normalize(x)

        cache = {
            "inputs": [x],
            "pre_activations": []
        }

        current = x

        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = current @ W + b
            cache["pre_activations"].append(z)

            if i < len(self.weights) - 1:
                current = self.relu(z)
            else:
                current = np.clip(z, -1.0, 1.0)

            cache["inputs"].append(current)

        return current, cache

    def predict(self, x):
        normalized_action, _ = self.forward(x)

        physical_action = self.denormalize_actions(normalized_action)

        physical_action[:, 0] = np.clip(
            physical_action[:, 0],
            0.0,
            self.max_v
        )

        physical_action[:, 1] = np.clip(
            physical_action[:, 1],
            -self.max_omega,
            self.max_omega
        )

        return physical_action

    def train_step(self, X_batch, Y_batch, current_lr):
        preds, cache = self.forward(
            X_batch,
            is_normalized=True
        )

        loss = np.mean((preds - Y_batch) ** 2)

        N = X_batch.shape[0]
        d_preds = 2.0 * (preds - Y_batch) / N

        z_last = cache["pre_activations"][-1]

        clip_grad = (
            (z_last > -1.0) &
            (z_last < 1.0)
        ).astype(np.float32)

        delta = d_preds * clip_grad

        grad_W = []
        grad_b = []

        for i in reversed(range(len(self.weights))):
            inp = cache["inputs"][i]

            gW = inp.T @ delta
            gb = np.sum(delta, axis=0, keepdims=True)

            gW = np.clip(gW, -5.0, 5.0)
            gb = np.clip(gb, -5.0, 5.0)

            grad_W.insert(0, gW)
            grad_b.insert(0, gb)

            if i > 0:
                delta = (
                    delta @ self.weights[i].T
                ) * self.relu_grad(
                    cache["pre_activations"][i - 1]
                )

        self.t += 1

        lr_t = (
            current_lr
            * np.sqrt(1.0 - 0.999 ** self.t)
            / (1.0 - 0.9 ** self.t)
        )

        for i in range(len(self.weights)):
            self.m_w[i] = (
                0.9 * self.m_w[i]
                + 0.1 * grad_W[i]
            )

            self.v_w[i] = (
                0.999 * self.v_w[i]
                + 0.001 * (grad_W[i] ** 2)
            )

            self.weights[i] -= (
                lr_t
                * self.m_w[i]
                / (np.sqrt(self.v_w[i]) + 1e-8)
            )

            self.m_b[i] = (
                0.9 * self.m_b[i]
                + 0.1 * grad_b[i]
            )

            self.v_b[i] = (
                0.999 * self.v_b[i]
                + 0.001 * (grad_b[i] ** 2)
            )

            self.biases[i] -= (
                lr_t
                * self.m_b[i]
                / (np.sqrt(self.v_b[i]) + 1e-8)
            )

        return loss

    def save_weights(self, path="imitation_policy_weights.npz"):
        arrays = {
            f"W{i}": w
            for i, w in enumerate(self.weights)
        }

        arrays.update({
            f"b{i}": b
            for i, b in enumerate(self.biases)
        })

        arrays["mean"] = self.mean
        arrays["std"] = self.std

        np.savez(path, **arrays)

        print(
            f"[INFO] Policy weights and normalization "
            f"parameters saved to '{path}'"
        )

    def load_weights(self, path="imitation_policy_weights.npz"):
        data = np.load(path)

        for i in range(len(self.weights)):
            self.weights[i] = data[f"W{i}"]
            self.biases[i] = data[f"b{i}"]

        if "mean" in data:
            self.mean = data["mean"]

        if "std" in data:
            self.std = data["std"]

        print(
            f"[INFO] Loaded weights and normalization "
            f"statistics from '{path}'"
        )


def train_policy():

    dataset_path = "demo_dataset.npz"

    if not os.path.exists(dataset_path):
        from collect_dataset import collect_demonstrations

        print(
            "[INFO] Demonstration dataset not found. "
            "Collecting now..."
        )

        collect_demonstrations()

    data = np.load(dataset_path)

    X = data["observations"].astype(np.float32)
    Y = data["actions"].astype(np.float32)

    num_samples = len(X)

    print("=" * 70)
    print("  Behavioral Cloning Pretraining for AIRL")
    print("=" * 70)
    print(f"  Dataset Samples : {num_samples}")
    print(f"  Observation Dim : {X.shape[1]}")
    print(f"  Action Dim      : {Y.shape[1]}")
    print("=" * 70)

    policy = ImitationPolicyNet(
        layer_sizes=[28, 128, 64, 2],
        lr=0.001,
        max_v=4.0,
        max_omega=2.5
    )

    policy.mean = np.mean(X, axis=0)
    policy.std = np.std(X, axis=0) + 1e-6

    X_norm = policy.normalize(X)

    Y_norm = policy.normalize_actions(Y)

    batch_size = 256
    epochs = 200

    for epoch in range(epochs):

        current_lr = 0.001 * (
            0.98 ** (epoch // 5)
        )

        indices = np.random.permutation(num_samples)

        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, num_samples, batch_size):

            batch_idx = indices[i:i + batch_size]

            X_b = X_norm[batch_idx]
            Y_b = Y_norm[batch_idx]

            loss = policy.train_step(
                X_b,
                Y_b,
                current_lr
            )

            epoch_loss += loss
            num_batches += 1

        avg_loss = epoch_loss / num_batches

        if (epoch + 1) % 15 == 0 or epoch == 0:

            print(
                f"  Epoch {epoch + 1:3d}/{epochs} | "
                f"LR: {current_lr:.6f} | "
                f"Normalized MSE: {avg_loss:.6f}"
            )

    policy.save_weights(
        "imitation_policy_weights.npz"
    )

    print("=" * 70)
    print("  BC Pretraining Complete")
    print("=" * 70)


if __name__ == "__main__":
    train_policy()