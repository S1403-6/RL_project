import numpy as np
import json
import os
from feature_extractor import CNNFeatureExtractor, SyntheticFrameGenerator

EPOCHS     = 100
SAMPLES    = 5000
BATCH_SIZE = 64
LR         = 0.001
INPUT_DIM  = 16

np.random.seed(42)

class NumpyNeuralNet:
    def __init__(self, layer_sizes=[16, 128, 64, 2], lr=0.001):
        self.lr = lr
        self.weights = []
        self.biases  = []

        for i in range(len(layer_sizes) - 1):
            fan_in  = layer_sizes[i]
            fan_out = layer_sizes[i+1]
            W = np.random.randn(fan_in, fan_out) * np.sqrt(2.0/fan_in)
            b = np.zeros((1, fan_out))
            self.weights.append(W)
            self.biases.append(b)

    def relu(self, x): return np.maximum(0, x)
    def relu_grad(self, x): return (x > 0).astype(float)

    def forward(self, x):
        x = x.reshape(1, -1)
        activations = [x]
        preacts = []

        current = x
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = current @ W + b
            preacts.append(z)
            if i < len(self.weights) - 1:
                current = self.relu(z)
            else:
                current = z   # linear output for regression
            activations.append(current)

        return current.flatten(), activations, preacts

    def backward(self, activations, preacts, target):
        pred = activations[-1]
        target = target.reshape(1, -1)

        # MSE loss gradient
        delta = (pred - target)

        grad_W, grad_b = [], []

        for i in reversed(range(len(self.weights))):
            a_prev = activations[i]

            grad_W.insert(0, a_prev.T @ delta)
            grad_b.insert(0, delta)

            if i > 0:
                delta = delta @ self.weights[i].T
                delta = delta * self.relu_grad(preacts[i-1])

        return grad_W, grad_b

    def update(self, grad_W, grad_b):
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * grad_W[i]
            self.biases[i]  -= self.lr * grad_b[i]

    def save(self, path="nn_weights.npz"):
        arrays = {f"W{i}": w for i, w in enumerate(self.weights)}
        arrays.update({f"b{i}": b for i, b in enumerate(self.biases)})
        np.savez(path, **arrays)

# -----------------------------
# EXPERT POLICY (VERY IMPORTANT)
# -----------------------------
def expert_policy(features):
    dist = features[0]
    angle = features[1]
    density = features[13]
    flow_mag = np.linalg.norm(features[14:16])

    # --- Kp logic ---
    if dist > 0.5:
        Kp = 3.0
    else:
        Kp = 1.5

    # reduce if obstacles
    Kp *= (1.0 - 0.5 * density)

    # --- Kd logic ---
    Kd = 1.0 + 2.0 * (1 - dist)   # higher near goal
    Kd += 1.5 * flow_mag          # damping for motion

    # clip
    Kp = np.clip(Kp, 0.5, 5.0)
    Kd = np.clip(Kd, 0.5, 5.0)

    return np.array([Kp, Kd], dtype=np.float32)

# -----------------------------
# DATA GENERATION
# -----------------------------
def generate_dataset(num_samples):
    extractor = CNNFeatureExtractor()
    gen = SyntheticFrameGenerator()

    X = []
    Y = []

    for _ in range(num_samples):
        robot_pos = np.random.uniform([-1, -1], [1, 1])
        robot_yaw = np.random.uniform(-np.pi, np.pi)

        target_pos = [6.0, 0.0]

        obs_positions = [
            [np.random.uniform(1,4), np.random.uniform(-2,2)]
            for _ in range(5)
        ]

        frame = gen.render(robot_pos, robot_yaw, target_pos, obs_positions)
        features = extractor.extract(frame)

        kp_kd = expert_policy(features)

        X.append(features)
        Y.append(kp_kd)

    return np.array(X), np.array(Y)


def train():
    net = NumpyNeuralNet(lr=LR)

    print("Generating dataset...")
    X, Y = generate_dataset(SAMPLES)

    print("Training started...")

    for epoch in range(EPOCHS):
        indices = np.random.permutation(len(X))

        total_loss = 0

        for i in range(0, len(X), BATCH_SIZE):
            batch_idx = indices[i:i+BATCH_SIZE]

            grad_W_sum = [np.zeros_like(w) for w in net.weights]
            grad_b_sum = [np.zeros_like(b) for b in net.biases]

            batch_loss = 0

            for idx in batch_idx:
                x = X[idx]
                y = Y[idx]

                pred, activations, preacts = net.forward(x)

                loss = np.mean((pred - y)**2)
                batch_loss += loss

                gW, gb = net.backward(activations, preacts, y)

                for j in range(len(net.weights)):
                    grad_W_sum[j] += gW[j]
                    grad_b_sum[j] += gb[j]

            # update
            for j in range(len(net.weights)):
                net.weights[j] -= LR * grad_W_sum[j] / len(batch_idx)
                net.biases[j]  -= LR * grad_b_sum[j] / len(batch_idx)

            total_loss += batch_loss

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f}")

    net.save("nn_weights2.npz")
    print("Training complete. Weights saved!")

# -----------------------------
if __name__ == "__main__":
    train()