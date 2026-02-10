"""
pytorch_intro.py — Introduction to PyTorch (Tensors, Autograd, Training, Checkpoints)

This script walks through a classic PyTorch mini-ML pipeline step-by-step and explains:
- What each step is
- Why we do it
- A clear example of input -> output

Run:
    python3 -m pip install torch
    python3 pytorch_intro.py

Note:
- CUDA works only with NVIDIA GPUs and a CUDA-enabled PyTorch build.
- On Mac, you typically use CPU or MPS (Apple Silicon) instead of CUDA.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------
# Pretty printing helpers (same spirit as the NLTK file)
# ---------------------------------------------------------
def title(s: str) -> None:
    print("\n" + "=" * 78)
    print(s)
    print("=" * 78)


def show_io(label_in: str, x_in, label_out: str, x_out) -> None:
    """Small helper: show input -> output in a consistent format."""
    print(f"\n{label_in}:")
    print(x_in)
    print(f"\n{label_out}:")
    print(x_out)


# =========================================================
# 1) TENSORS
# =========================================================
def demo_tensors() -> None:
    """
    Tensors are PyTorch’s core data structure.

    What is a tensor?
    - A tensor is a multi-dimensional numeric container.
    - It generalizes:
        scalar (0D), vector (1D), matrix (2D), and higher dimensions (3D+).
    - A tensor has 3 key attributes:
        1) shape  -> dimensions (ex: [3,4])
        2) dtype  -> number type (int64, float32, ...)
        3) device -> where it lives (cpu, cuda)

    Why it matters:
    - Every neural network input/output is a tensor.
    - Most errors in PyTorch come from shape/dtype/device mismatches.

    This demo shows:
    - Creating a 1D tensor (vector)
    - Creating a 2D tensor (matrix)
    - Reading .shape and .dtype
    """
    title("1) TENSORS: creation, shape, dtype")

    t1 = torch.tensor([1, 2, 3, 4])
    show_io("Input (Python list)", [1, 2, 3, 4], "Output (torch.tensor)", t1)
    show_io("t1.shape", t1.shape, "t1.dtype", t1.dtype)

    t2 = torch.tensor([[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12]])
    show_io("Input (list of lists)", "[[1,2,3,4],[5,6,7,8],[9,10,11,12]]", "Output (tensor)", t2)
    show_io("t2.shape", t2.shape, "Interpretation", "3 rows × 4 columns")

    print("\nKey idea:")
    print("- t2.shape = (3,4) means: 3 rows, 4 columns.")
    print("- The first number is rows, the second is columns (for 2D tensors).")


# =========================================================
# 2) COMMON CREATION ERRORS
# =========================================================
def demo_tensor_creation_errors() -> None:
    """
    Common tensor creation errors.

    A) Ragged rows (non-rectangular data)
    - A 2D tensor is like a matrix.
    - A matrix must be rectangular: every row has the same number of columns.
    - If one row has 5 elements and another has 4, PyTorch refuses.

    B) Mixed types (strings + numbers)
    - Tensors are numeric (used for math).
    - Mixing strings and numbers prevents PyTorch from picking a numeric dtype.
    - NumPy sometimes converts everything to strings, but that’s not useful for ML.

    This demo reproduces your errors exactly and explains them.
    """
    title("2) COMMON ERRORS: ragged rows and mixed types")

    # A) Ragged rows
    try:
        t3 = torch.tensor([[1, 2, 3, 4],
                           [5, 6, 7, 8, 111],
                           [9, 10, 11, 12]])
        show_io("Input (ragged list)", "row lengths: 4, 5, 4", "Output", t3)
    except Exception as e:
        show_io("Input (ragged list)", "row lengths: 4, 5, 4", "Output (error)", e)

    print("\nWhy this error happens:")
    print("- PyTorch tries to build a 3×4 tensor, but the second row is length 5.")
    print("- There is no single rectangular shape that fits all rows.")

    # B) Mixed types
    a = [["aa", 1, 2, 3],
         [5, 6, 7, 8],
         [9, 10, 11, 12]]
    try:
        ta = torch.tensor(a)
        show_io("Input (mixed types)", a, "Output", ta)
    except Exception as e:
        show_io("Input (mixed types)", a, "Output (error)", e)

    print("\nWhy this error happens:")
    print("- A tensor must have ONE dtype (example: float32).")
    print("- A string 'aa' cannot be stored in a numeric tensor.")


# =========================================================
# 3) RESHAPING
# =========================================================
def demo_reshaping() -> None:
    """
    Reshaping changes how the SAME elements are grouped into dimensions.

    Key rules:
    1) Reshape cannot change the number of elements.
       Example: 3×4 = 12 elements can become 2×6 = 12 elements.
    2) reshape() is NOT in-place.
       It returns a new tensor with a new view of the data.
       The old tensor stays the same unless you reassign.

    This demo shows:
    - Original tensor shape (3×4)
    - Reshaped tensor shape (2×6)
    - Original tensor remains (3×4)
    """
    title("3) RESHAPING: reshape() returns a new tensor")

    t2 = torch.tensor([[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12]])

    show_io("Original t2", t2, "Original t2.shape", t2.shape)

    t2r = t2.reshape(2, 6)
    show_io("Input", "t2.reshape(2,6)", "Output (reshaped)", t2r)
    show_io("t2r.shape", t2r.shape, "Original t2.shape (unchanged)", t2.shape)

    print("\nKey idea:")
    print("- reshape creates a new object (t2r).")
    print("- If you want to keep it, you must do: t2 = t2.reshape(2,6)")


# =========================================================
# 4) DEVICES (CPU vs GPU / CUDA)
# =========================================================
def demo_devices() -> torch.device:
    """
    Devices: where tensors live and where computations happen.

    Device options:
    - cpu  : always available, runs on the processor
    - cuda : runs on an NVIDIA GPU (only if you have CUDA available)

    Why it matters:
    - GPUs are much faster for large matrix operations (deep learning training).
    - But CUDA is not always available, so code should fall back to CPU.

    Best practice:
    - Pick your device once:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    - Move tensors or models to that device.

    This demo shows:
    - Checking CUDA availability
    - Selecting a device
    - Moving a tensor to the chosen device
    """
    title("4) DEVICES: CPU vs GPU (CUDA)")

    cuda_ok = torch.cuda.is_available()
    show_io("torch.cuda.is_available()", cuda_ok, "Meaning", "True => CUDA usable, False => CPU only")

    device = torch.device("cuda") if cuda_ok else torch.device("cpu")
    show_io("Selected device", device, "Why we do this", "Same code works on CPU or GPU")

    t = torch.rand(2, 3)
    show_io("Random tensor device (default)", t.device, "After .to(device)", t.to(device).device)

    return device


# =========================================================
# 5) AUTOGRAD
# =========================================================
def demo_autograd() -> None:
    """
    Autograd: automatic gradient computation.

    What is a gradient?
    - A gradient tells you how a value changes when you change an input.
    - In ML, gradients tell us how to change weights to reduce the loss.

    How Autograd works:
    - requires_grad=True means: “track operations for differentiation”
    - PyTorch builds a computation graph as you do operations.
    - backward() computes gradients using the chain rule.
    - After backward(), gradients are stored in .grad

    This demo reproduces your example:
        e = (a*b - 5)^2
    Then computes:
        de/da and de/db
    """
    title("5) AUTOGRAD: requires_grad and backward()")

    a = torch.tensor(3.0, requires_grad=True)
    b = torch.tensor(4.0, requires_grad=True)

    c = a * b
    d = c - 5
    e = d ** 2

    show_io("Inputs (a, b)", (a.item(), b.item()), "Output e", e.item())

    e.backward()
    show_io("Gradient de/da", a.grad.item(), "Gradient de/db", b.grad.item())

    print("\nInterpretation:")
    print("- de/da = 56 means: if a increases slightly, e increases ~56×that amount (locally).")
    print("- de/db = 42 means: if b increases slightly, e increases ~42×that amount (locally).")


# =========================================================
# 6) TRAINING (mini loop)
# =========================================================
def demo_training(device: torch.device) -> Tuple[nn.Module, optim.Optimizer, List[float], List[float]]:
    """
    Training: learning parameters (weights) that reduce the loss.

    The training loop always follows the same pattern:
        1) Forward pass   -> compute predictions
        2) Loss           -> measure how wrong we are
        3) Backward pass  -> compute gradients of loss wrt weights
        4) Update step    -> adjust weights to reduce loss

    We train a tiny model:
        y_pred = w * x

    Two versions:
    A) Manual gradient descent:
       - You update w yourself: w = w - lr * grad
    B) nn.Module + optimizer:
       - Standard PyTorch way (scales to big networks)
       - optimizer.step() updates parameters automatically

    Output:
    - model, optimizer, and lists (losses, weights) for tracking
    """
    title("6) TRAINING: manual loop and nn.Module + optimizer")

    X = torch.tensor([[7.01], [3.02], [4.99], [8.00]], dtype=torch.float32, device=device)
    Y = torch.tensor([[14.01], [6.01], [10.00], [16.04]], dtype=torch.float32, device=device)

    # -------------------------
    # A) Manual gradient descent
    # -------------------------
    title("6A) TRAINING (manual): y_pred = w*x")

    w = torch.tensor([1.0], dtype=torch.float32, requires_grad=True, device=device)
    lr = 0.001

    for epoch in range(10):
        y_pred = w * X
        loss = ((y_pred - Y) ** 2).mean()
        loss.backward()

        with torch.no_grad():
            w -= lr * w.grad
            w.grad.zero_()

        print(f"Epoch {epoch:02d}: Loss={loss.item():.6f}, w={w.item():.6f}")

    # -------------------------
    # B) nn.Module + optimizer
    # -------------------------
    title("6B) TRAINING (PyTorch style): nn.Module + optimizer")

    class MyMulModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))

        def forward(self, x):
            return self.w * x

    model = MyMulModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    losses: List[float] = []
    weights: List[float] = []

    for epoch in range(10):
        y_pred = model(X)
        loss = criterion(y_pred, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        weights.append(model.w.item())

        print(f"Epoch {epoch:02d}: Loss={loss.item():.6f}, w={model.w.item():.6f}")

    return model, optimizer, losses, weights


# =========================================================
# 7) CHECKPOINTS + TRACKING
# =========================================================
def demo_checkpoints_and_tracking(
    model: nn.Module,
    optimizer: optim.Optimizer,
    losses: List[float],
    weights: List[float],
    device: torch.device,
    path: str = "mycheckpoint.pth",
) -> None:
    """
    CHECKPOINTS (saving & loading)

    What is a checkpoint?
    - A checkpoint is a saved snapshot of training so you can restart later.
    - Think: “Save game” in a video game.

    Why we save:
    - To resume training if it stops
    - To reuse the trained model later for predictions (inference)

    What we usually save:
    - model_state_dict: the learned weights
    - optimizer_state_dict: optimizer internal state (important to resume smoothly)
    - epoch: where we stopped
    - loss: last loss (optional)

    TRACKING TRAINING

    What is tracking?
    - Tracking means storing numbers (loss, weights) at each epoch.
    - This helps you see if training improves over time.

    Tools:
    - Matplotlib: plot loss curves
    - TensorBoard: real-time dashboards and experiment comparison

    This demo:
    - Saves a checkpoint
    - Loads it back
    - Restores model/optimizer states
    - Prints code snippets for Matplotlib and TensorBoard
    """
    title("7) CHECKPOINTS + TRACKING: save/load + visualize metrics")

    # ---- Save
    torch.save(
        {
            "epoch": len(losses) - 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": losses[-1],
        },
        path,
    )
    show_io("Saved checkpoint path", path, "Saved keys", ["epoch", "model_state_dict", "optimizer_state_dict", "loss"])

    # ---- Load
    chkpnt = torch.load(path, map_location=device)
    show_io("Loaded checkpoint keys", list(chkpnt.keys()), "Loaded epoch", chkpnt["epoch"])

    # Restore into a new model/optimizer
    class MyMulModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))

        def forward(self, x):
            return self.w * x

    restored_model = MyMulModel().to(device)
    restored_optimizer = optim.SGD(restored_model.parameters(), lr=0.001)

    restored_model.load_state_dict(chkpnt["model_state_dict"])
    restored_optimizer.load_state_dict(chkpnt["optimizer_state_dict"])

    show_io("Restored w", restored_model.w.item(), "Last saved loss", chkpnt["loss"])

    print("\nTracking (Matplotlib) snippet:")
    print("""
# import matplotlib.pyplot as plt
# plt.plot(losses, label="Loss")
# plt.plot(weights, label="Weight (w)")
# plt.legend()
# plt.show()
""")

    print("Tracking (TensorBoard) snippet:")
    print("""
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter("runs/MyMulModel_experiment")
# for epoch, (l, w) in enumerate(zip(losses, weights)):
#     writer.add_scalar("Loss", l, epoch)
#     writer.add_scalar("Weight/w", w, epoch)
# writer.close()
# tensorboard --logdir runs
""")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main() -> None:

    demo_tensors()
    demo_tensor_creation_errors()
    demo_reshaping()
    device = demo_devices()
    demo_autograd()
    model, optimizer, losses, weights = demo_training(device)
    demo_checkpoints_and_tracking(model, optimizer, losses, weights, device)

    title("Done ✅")
    print("PyTorch demo finished.")


if __name__ == "__main__":
    main()