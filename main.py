
import csv
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load Iris Dataset
# -------------------------------
X = []
y = []

with open("iris.csv", "r") as file:
    reader = csv.DictReader(file)

    for row in reader:
        species = row["species"]

        if species in ["setosa", "versicolor"]:
            X.append([
                float(row["petal_length"]),
                float(row["petal_width"])
            ])

            y.append(-1 if species == "setosa" else 1)

X = np.array(X)
y = np.array(y)

print("Dataset shape:", X.shape)
print("Labels shape:", y.shape)


# -------------------------------
# 2. Perceptron Class
# -------------------------------
class Perceptron:

    def __init__(self, learning_rate=1.0):
        self.lr = learning_rate
        self.w = np.zeros(2)
        self.b = 0.0

        # Store initial weights and bias
        self.weight_history = [self.w.copy()]
        self.bias_history = [self.b]

    def predict(self, x):
        score = np.dot(self.w, x) + self.b
        return 1 if score >= 0 else -1

    # -------------------------------
    # 3. Online Perceptron
    # -------------------------------
    def fit_online(self, X, y, epochs=100):
        print("\n--- ONLINE PERCEPTRON ---")

        for epoch in range(epochs):
            updates = 0

            for i in range(len(X)):
                prediction = self.predict(X[i])

                if prediction != y[i]:
                    self.w += self.lr * y[i] * X[i]
                    self.b += self.lr * y[i]
                    updates += 1

            self.weight_history.append(self.w.copy())
            self.bias_history.append(self.b)

            print(
                f"Online Epoch {epoch + 1}: "
                f"Updates = {updates}, "
                f"Weights = {self.w}, Bias = {self.b}"
            )

            if updates == 0:
                print("Online: Converged!")
                break

        print("Online final weights:", self.w)
        print("Online final bias:", self.b)

    # -------------------------------
    # 4. Full-Epoch Perceptron
    # -------------------------------
    def fit_full_epoch(self, X, y, epochs=100):
        print("\n--- FULL-EPOCH PERCEPTRON ---")

        for epoch in range(epochs):
            total_w = np.zeros_like(self.w)
            total_b = 0.0
            mistakes = 0

            for i in range(len(X)):
                prediction = self.predict(X[i])

                if prediction != y[i]:
                    total_w += self.lr * y[i] * X[i]
                    total_b += self.lr * y[i]
                    mistakes += 1

            # Apply the accumulated updates together
            self.w += total_w
            self.b += total_b

            self.weight_history.append(self.w.copy())
            self.bias_history.append(self.b)

            print(
                f"Full Epoch {epoch + 1}: "
                f"Misclassified = {mistakes}, "
                f"Weights = {self.w}, Bias = {self.b}"
            )

            if mistakes == 0:
                print("Full-epoch: Converged!")
                break

        print("Full-epoch final weights:", self.w)
        print("Full-epoch final bias:", self.b)


# -------------------------------
# 5. Train Both Models
# -------------------------------
online_model = Perceptron()
online_model.fit_online(X, y)

epoch_model = Perceptron()
epoch_model.fit_full_epoch(X, y)


# -------------------------------
# 6. Plot Weight History
# -------------------------------
def plot_weight_history(online_model, epoch_model):

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    models = [
        ("Online Perceptron", online_model),
        ("Full-Epoch Perceptron", epoch_model)
    ]

    settings = [
        (0, "Weight w1"),
        (1, "Weight w2"),
        (2, "Bias b")
    ]

    for ax, (index, title) in zip(axes, settings):
        for name, model in models:

            if index == 0:
                values = [w[0] for w in model.weight_history]
            elif index == 1:
                values = [w[1] for w in model.weight_history]
            else:
                values = model.bias_history

            ax.plot(
                range(len(values)),
                values,
                marker="o",
                label=name
            )

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.savefig("weight_history.png", dpi=300)
    print("\nSaved graph as weight_history.png")
    plt.show()


plot_weight_history(online_model, epoch_model)

# -------------------------------
# 7. Plot Decision Boundary
# -------------------------------
def plot_decision_boundary(X, y, model, title):

    plt.figure(figsize=(8, 6))

    # Plot both classes
    plt.scatter(
        X[y == -1, 0],
        X[y == -1, 1],
        color="blue",
        label="Setosa"
    )

    plt.scatter(
        X[y == 1, 0],
        X[y == 1, 1],
        color="red",
        label="Versicolor"
    )

    # Decision boundary: w1*x1 + w2*x2 + b = 0
    w1, w2 = model.w

    if abs(w2) > 1e-10:
        x_values = np.linspace(X[:, 0].min(), X[:, 0].max(), 200)
        y_values = -(w1 * x_values + model.b) / w2

        plt.plot(
            x_values,
            y_values,
            color="green",
            label="Decision Boundary"
        )

    elif abs(w1) > 1e-10:
        x_boundary = -model.b / w1
        plt.axvline(
            x=x_boundary,
            color="green",
            label="Decision Boundary"
        )

    plt.xlabel("Petal Length (cm)")
    plt.ylabel("Petal Width (cm)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    filename = title.lower().replace(" ", "_") + ".png"
    plt.savefig(filename, dpi=300)
    print("Saved:", filename)
    plt.show()


# Plot boundaries for both trained models
plot_decision_boundary(
    X, y, online_model,
    "Online Perceptron Decision Boundary"
)

plot_decision_boundary(
    X, y, epoch_model,
    "Full Epoch Perceptron Decision Boundary"
)

# -------------------------------
# 8. Final Decision Boundary with Margins
# -------------------------------
def plot_final_boundary_with_margin(X, y, model):

    w1, w2 = model.w
    norm_w = np.linalg.norm(model.w)

    if norm_w == 0:
        print("Cannot plot boundary: weights are zero.")
        return

    plt.figure(figsize=(8, 6))

    # Plot Setosa and Versicolor
    plt.scatter(
        X[y == -1, 0], X[y == -1, 1],
        color="blue", label="Setosa"
    )

    plt.scatter(
        X[y == 1, 0], X[y == 1, 1],
        color="red", label="Versicolor"
    )

    x_values = np.linspace(
        X[:, 0].min() - 0.2,
        X[:, 0].max() + 0.2,
        200
    )

    if abs(w2) > 1e-10:
        # Decision boundary
        boundary = -(w1 * x_values + model.b) / w2

        plt.plot(
            x_values, boundary,
            color="green", label="Decision Boundary"
        )

        # Reference margin lines
        margin_positive = (1 - model.b - w1 * x_values) / w2
        margin_negative = (-1 - model.b - w1 * x_values) / w2

        plt.plot(
            x_values, margin_positive,
            "k--", label="Reference Margin (+1)"
        )

        plt.plot(
            x_values, margin_negative,
            "k--", label="Reference Margin (-1)"
        )

    elif abs(w1) > 1e-10:
        boundary_x = -model.b / w1
        plt.axvline(
            boundary_x, color="green",
            label="Decision Boundary"
        )

        plt.axvline(
            (1 - model.b) / w1,
            color="black", linestyle="--",
            label="Reference Margin (+1)"
        )

        plt.axvline(
            (-1 - model.b) / w1,
            color="black", linestyle="--",
            label="Reference Margin (-1)"
        )

    plt.xlabel("Petal Length (cm)")
    plt.ylabel("Petal Width (cm)")
    plt.title("Final Decision Boundary with Margins")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("final_boundary_with_margin.png", dpi=300)
    print("Figure 7 saved as final_boundary_with_margin.png")

    plt.show()


# Generate Figure 7
plot_final_boundary_with_margin(X, y, online_model)

# -------------------------------
# Task 2 - Step 1: Perceptron Criterion Loss
# -------------------------------

def perceptron_loss(X, y, w, b):
    scores = np.dot(X, w) + b
    sample_losses = np.maximum(0, -y * scores)
    return np.mean(sample_losses)


# Evaluate loss using zero-initialized parameters
initial_w = np.zeros(2)
initial_b = 0.0

initial_loss = perceptron_loss(
    X, y, initial_w, initial_b
)

print("\n--- TASK 2: LOSS FUNCTION ---")
print("Loss function: L = max(0, -y * (w.x + b))")
print("Initial weights:", initial_w)
print("Initial bias:", initial_b)
print("Initial mean loss:", initial_loss)

# =====================================================
# TASK 2: GRADIENT-DESCENT PERCEPTRON
# =====================================================

def train_gradient_descent(
    X, y, method="batch", lr=0.01,
    epochs=100, batch_size=10
):
    w = np.zeros(X.shape[1])
    b = 0.0

    losses = []
    boundary_history = []
    update_count = 0
    rng = np.random.default_rng(42)

    for epoch in range(epochs):
        if method == "batch":
            batches = [np.arange(len(X))]
        elif method == "mini-batch":
            order = rng.permutation(len(X))
            batches = [
                order[i:i + batch_size]
                for i in range(0, len(X), batch_size)
            ]
        else:  # SGD: one sample per update
            order = rng.permutation(len(X))
            batches = [np.array([i]) for i in order]

        for indices in batches:
            xb = X[indices]
            yb = y[indices]

            scores = xb @ w + b
            signed_scores = yb * scores

            # Perceptron loss subgradient.
            # At score zero, choose the active subgradient.
            active = signed_scores <= 0

            if np.any(active):
                grad_w = -np.mean(
                    yb[active, None] * xb[active], axis=0
                ) * (np.sum(active) / len(indices))

                grad_b = -np.sum(yb[active]) / len(indices)

                w -= lr * grad_w
                b -= lr * grad_b
                update_count += 1

        losses.append(perceptron_loss(X, y, w, b))
        boundary_history.append((w.copy(), b))

    predictions = np.where(X @ w + b >= 0, 1, -1)
    mistakes = int(np.sum(predictions != y))

    return {
        "method": method,
        "w": w,
        "b": b,
        "losses": losses,
        "history": boundary_history,
        "updates": update_count,
        "mistakes": mistakes,
        "epochs": len(losses),
        "final_loss": losses[-1]
    }


# -------------------------------
# Train all three variants
# -------------------------------
gd_models = []

for method in ["batch", "mini-batch", "sgd"]:
    model = train_gradient_descent(
        X, y,
        method=method,
        lr=0.01,
        epochs=100,
        batch_size=10
    )
    gd_models.append(model)

    print(f"\n--- {method.upper()} GRADIENT DESCENT ---")
    print("Learning rate: 0.01")
    print("Batch size:", 100 if method == "batch"
          else 10 if method == "mini-batch" else 1)
    print("Epochs:", model["epochs"])
    print("Training mistakes:", model["mistakes"])
    print("Update count:", model["updates"])
    print("Final weights:", model["w"])
    print("Final bias:", model["b"])
    print("Final loss:", model["final_loss"])


# -------------------------------
# Figure 8: Loss curves
# -------------------------------
plt.figure(figsize=(9, 6))

for model in gd_models:
    plt.plot(
        range(1, len(model["losses"]) + 1),
        model["losses"],
        label=model["method"]
    )

plt.xlabel("Epoch")
plt.ylabel("Mean Training Loss")
plt.title("Gradient Descent: Loss vs Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("gd_loss_curve.png", dpi=300)
plt.show()

print("Figure 8 saved as gd_loss_curve.png")


# -------------------------------
# Helper: draw a decision boundary
# -------------------------------
def draw_boundary(ax, X, y, w, b, title):
    ax.scatter(X[y == -1, 0], X[y == -1, 1],
               label="Setosa")
    ax.scatter(X[y == 1, 0], X[y == 1, 1],
               label="Versicolor")

    if abs(w[1]) > 1e-10:
        xs = np.linspace(X[:, 0].min(), X[:, 0].max(), 200)
        ys = -(w[0] * xs + b) / w[1]
        ax.plot(xs, ys, label="Boundary")
    elif abs(w[0]) > 1e-10:
        ax.axvline(-b / w[0], label="Boundary")

    ax.set_xlabel("Petal Length (cm)")
    ax.set_ylabel("Petal Width (cm)")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()


# -------------------------------
# Figure 9: Boundary evolution
# -------------------------------
batch_model = gd_models[0]
total_epochs = batch_model["epochs"]

selected_epochs = sorted(set([
    1,
    max(1, total_epochs // 3),
    max(1, 2 * total_epochs // 3),
    total_epochs
]))

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

for ax, epoch_number in zip(axes.flat, selected_epochs):
    w, b = batch_model["history"][epoch_number - 1]
    draw_boundary(
        ax, X, y, w, b,
        f"Batch GD - Epoch {epoch_number}"
    )

plt.tight_layout()
plt.savefig("gd_boundary_evolution.png", dpi=300)
plt.show()

print("Figure 9 saved as gd_boundary_evolution.png")


# -------------------------------
# Final boundary comparison
# -------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

draw_boundary(
    axes[0], X, y,
    online_model.w, online_model.b,
    "Classic Online Perceptron"
)

draw_boundary(
    axes[1], X, y,
    batch_model["w"], batch_model["b"],
    "Final Batch Gradient Descent"
)

plt.tight_layout()
plt.savefig("gd_final_boundary_comparison.png", dpi=300)
plt.show()

print("Final boundary comparison saved.")


# -------------------------------
# Comparison table
# -------------------------------
print("\n--- TASK 2: COMPARISON TABLE ---")
print(
    f"{'Method':<18} {'Mistakes':<10} {'Updates':<10} "
    f"{'Epochs':<8} {'Weights':<24} {'Bias':<12} {'Loss'}"
)

for model in gd_models:
    print(
        f"{model['method']:<18} "
        f"{model['mistakes']:<10} "
        f"{model['updates']:<10} "
        f"{model['epochs']:<8} "
        f"{np.array2string(model['w'], precision=4):<24} "
        f"{model['b']:<12.4f} "
        f"{model['final_loss']:.6f}"
    )

# Include the classic model using measured final mistakes
classic_predictions = np.where(
    X @ online_model.w + online_model.b >= 0, 1, -1
)
classic_mistakes = int(np.sum(classic_predictions != y))

print(
    f"{'Classic Online':<18} "
    f"{classic_mistakes:<10} "
    f"{'See epoch log':<10} "
    f"{len(online_model.weight_history) - 1:<8} "
    f"{np.array2string(online_model.w, precision=4):<24} "
    f"{online_model.b:<12.4f} "
    f"{perceptron_loss(X, y, online_model.w, online_model.b):.6f}"
)
# SECTION 6: EXPERIMENT 1 - LEARNING RATE
print("\n--- EXPERIMENT 1: LEARNING RATE ---")

for lr in [0.01, 0.1, 1.0, 10.0]:
    model = train_gradient_descent(
        X, y,
        method="batch",
        lr=lr,
        epochs=100,
        batch_size=10
    )

    print(f"\nLearning rate: {lr}")
    print("Epochs:", model["epochs"])
    print("Mistakes:", model["mistakes"])
    print("Updates:", model["updates"])
    print("Final weights:", model["w"])
    print("Final bias:", model["b"])
    print("Final loss:", model["final_loss"])
    
# EXPERIMENT 2: PRESENTATION ORDER
print("\n--- EXPERIMENT 2: PRESENTATION ORDER ---")

def run_order_experiment(X, y, shuffle=False, seed=42, epochs=100):
    model = Perceptron(learning_rate=1.0)
    rng = np.random.default_rng(seed)
    total_updates = 0
    converged = False

    for epoch in range(epochs):
        if shuffle:
            order = rng.permutation(len(X))
        else:
            order = np.arange(len(X))

        epoch_updates = 0

        for i in order:
            prediction = model.predict(X[i])

            if prediction != y[i]:
                model.w += model.lr * y[i] * X[i]
                model.b += model.lr * y[i]
                epoch_updates += 1

        total_updates += epoch_updates

        if epoch_updates == 0:
            converged = True
            break

    predictions = np.where(
        X @ model.w + model.b >= 0, 1, -1
    )
    mistakes = int(np.sum(predictions != y))

    return model, total_updates, epoch + 1, mistakes, converged


for shuffled in [False, True]:
    model, updates, epochs_used, mistakes, converged = (
        run_order_experiment(X, y, shuffle=shuffled, seed=42)
    )

    print("\nOrder:", "Shuffled" if shuffled else "Fixed")
    print("Seed: 42")
    print("Epochs:", epochs_used)
    print("Updates:", updates)
    print("Mistakes:", mistakes)
    print("Converged:", converged)
    print("Final weights:", model.w)
    print("Final bias:", model.b)

# EXPERIMENT 3: INITIALIZATION
print("\n--- EXPERIMENT 3: INITIALIZATION ---")

def run_initialization_experiment(X, y, random_init=False, seed=42, epochs=100):
    model = Perceptron(learning_rate=1.0)

    if random_init:
        rng = np.random.default_rng(seed)
        model.w = rng.uniform(-0.01, 0.01, size=X.shape[1])
        model.b = float(rng.uniform(-0.01, 0.01))

    print("Initial weights:", model.w)
    print("Initial bias:", model.b)

    updates = 0
    converged = False

    for epoch in range(epochs):
        epoch_updates = 0

        for i in range(len(X)):
            prediction = model.predict(X[i])

            if prediction != y[i]:
                model.w += model.lr * y[i] * X[i]
                model.b += model.lr * y[i]
                epoch_updates += 1

        updates += epoch_updates

        if epoch_updates == 0:
            converged = True
            break

    predictions = np.where(X @ model.w + model.b >= 0, 1, -1)
    mistakes = int(np.sum(predictions != y))

    print("Initialization:", "Small random" if random_init else "Zero")
    print("Seed:", seed if random_init else "Not applicable")
    print("Epochs:", epoch + 1)
    print("Updates:", updates)
    print("Mistakes:", mistakes)
    print("Converged:", converged)
    print("Final weights:", model.w)
    print("Final bias:", model.b)


run_initialization_experiment(X, y, random_init=False)
run_initialization_experiment(X, y, random_init=True, seed=42)
