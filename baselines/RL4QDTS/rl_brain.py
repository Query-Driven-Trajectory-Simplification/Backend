import numpy as np
import random
from collections import deque

# The two DQN agents of RL4QDTS. The original built them with Keras; this is the same network,
# loss and optimiser written in numpy (a network this small is ~100x faster without a framework):
#   state -> Dense(25) -> tanh -> Dense(action_size), Huber loss (delta 1) averaged over the
#   outputs, Adam (lr 0.01, beta1 0.9, beta2 0.999, eps 1e-7, as Keras/TF), float32 weights,
#   and replay() still fits the sampled transitions one at a time (batch size 1), as the original.
# Weights are saved in the Keras 2.2 .h5 layout, so files load in the original code and vice versa.

#for reproducible
np.random.seed(1)

LEGACY_TANH = False  # True: fast_online_act uses the original tanh, which returns NaN for |x| > ~710


class QNetwork:
    def __init__(self, state_size, action_size, learning_rate, rng, hidden=25):
        def glorot(n_in, n_out):  # Keras' default kernel initialiser
            limit = np.sqrt(6.0 / (n_in + n_out))
            return rng.uniform(-limit, limit, (n_in, n_out)).astype(np.float32)
        self.weights = [glorot(state_size, hidden), np.zeros(hidden, np.float32),
                        glorot(hidden, action_size), np.zeros(action_size, np.float32)]
        self.lr, self.beta1, self.beta2, self.eps = np.float32(learning_rate), 0.9, 0.999, np.float32(1e-7)
        self.m = [np.zeros_like(w) for w in self.weights]
        self.v = [np.zeros_like(w) for w in self.weights]
        self.iterations = 0

    def predict(self, state):
        w1, b1, w2, b2 = self.weights
        hidden = np.tanh(np.asarray(state, np.float32) @ w1 + b1)
        return hidden @ w2 + b2

    def fit_one(self, state, action, target):
        """One Adam step on one transition: the original model.fit(state, target) where target
        equals the prediction everywhere except at `action`, so only that output has an error"""
        w1, b1, w2, b2 = self.weights
        x = np.asarray(state, np.float32).reshape(1, -1)
        hidden = np.tanh(x @ w1 + b1)
        q = hidden @ w2 + b2
        # d/dq of mean(huber(target - q)) over the outputs
        g = np.float32(np.clip(q[0, action] - np.float32(target), -1.0, 1.0) / q.shape[1])
        d_out = np.zeros_like(q)
        d_out[0, action] = g
        d_hidden = (d_out @ w2.T) * (1 - hidden * hidden)
        grads = [x.T @ d_hidden, d_hidden[0], hidden.T @ d_out, d_out[0]]
        self.iterations += 1
        t = self.iterations
        alpha = self.lr * np.float32(np.sqrt(1 - self.beta2 ** t) / (1 - self.beta1 ** t))
        for w, m, v, grad in zip(self.weights, self.m, self.v, grads):
            m += (grad - m) * np.float32(1 - self.beta1)
            v += (grad * grad - v) * np.float32(1 - self.beta2)
            w -= (m * alpha) / (np.sqrt(v) + self.eps)

    def get_weights(self):
        return [w.copy() for w in self.weights]

    def set_weights(self, weights):
        self.weights = [np.array(w, np.float32) for w in weights]

    def save_weights(self, name):
        import h5py
        with h5py.File(name, "w") as f:
            f.attrs["backend"] = b"tensorflow"
            f.attrs["keras_version"] = b"2.2.0"
            f.attrs["layer_names"] = np.array([b"dense_1", b"activation_1", b"dense_2"])
            f.create_group("activation_1").attrs["weight_names"] = np.array([], np.float64)
            for layer, (kernel, bias) in (("dense_1", self.weights[:2]), ("dense_2", self.weights[2:])):
                g = f.create_group(layer)
                g.attrs["weight_names"] = np.array([f"{layer}/kernel:0".encode(), f"{layer}/bias:0".encode()])
                g.create_dataset(f"{layer}/kernel:0", data=kernel)
                g.create_dataset(f"{layer}/bias:0", data=bias)

    def load_weights(self, name):
        """Reads Keras .h5 weight files (Keras 2.2 as shipped by the authors, or later versions)"""
        import h5py
        weights = []
        with h5py.File(name, "r") as f:
            for layer in f.attrs["layer_names"]:
                group = f[layer.decode() if isinstance(layer, bytes) else layer]
                for wname in group.attrs.get("weight_names", []):
                    weights.append(np.array(group[wname.decode() if isinstance(wname, bytes) else wname]))
        self.set_weights(weights)


class _Agent:
    def __init__(self, state_size, action_size, seed=None):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99
        self.learning_rate = 0.01
        rng = np.random.RandomState(seed) if seed is not None else np.random
        self.model = QNetwork(state_size, action_size, self.learning_rate, rng)
        self.target_model = QNetwork(state_size, action_size, self.learning_rate, rng)
        self.update_target_model()

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def online_act(self, state):
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def relu(self, x):
        return np.maximum(0,x)

    def sigmoid(self, x):
        s = 1 / (1 + np.exp(-x))
        return s

    def tanh(self, x):
        if LEGACY_TANH:
            return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return np.tanh(x)

    def get_weights(self):
        self.w1, self.b1, self.w2, self.b2 = self.model.get_weights()

    def fast_online_act(self, state):
        #print('fast implementation with numpy matrix')
        o1 = self.tanh(np.dot(state, self.w1)+self.b1)
        o2 = np.dot(o1, self.w2)+self.b2
        return np.argmax(o2[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            # the original predicted the full target vector and changed only `action`
            if done:
                target = reward
            else:
                t = self.target_model.predict(next_state)[0]
                target = reward + self.gamma * np.amax(t)
            self.model.fit_one(state, action, target)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)
        self.get_weights()

    def save(self, name):
        self.model.save_weights(name)


class AGENT_Sel(_Agent):
    """Agent-Point: picks which of the K candidate points of a cube to insert"""


class AGENT_Sam(_Agent):
    """Agent-Cube: zooms into one of the 8 child cubes, or stops (action 8)"""
