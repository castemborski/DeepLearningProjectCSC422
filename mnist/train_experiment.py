"""
CSC422 Project Goal 1 — MNIST CNN Hyperparameter Experiments

Trains the ConvNet from main.py under different batch sizes and learning rates,
tracks per-epoch metrics, saves plots, and saves the best model.

Usage:
    python train_experiment.py [--epochs N]

Outputs (written next to this script):
    batch_size_experiment.png  — loss/accuracy curves for 4 batch sizes
    lr_experiment.png          — loss/accuracy curves for 4 learning rates
    mnist_cnn_best.pt          — state_dict of the best model found
"""

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use('Agg')  # must precede pyplot import for headless environments
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

# Add this file's directory to sys.path so "from main import Net" works
# regardless of the working directory the user runs from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import Net  # noqa: E402  (import after sys.path manipulation)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BATCH_SIZES = [32, 64, 128, 256]
LEARNING_RATES = [0.1, 0.5, 1.0, 2.0]
FIXED_LR = 1.0
FIXED_BATCH = 64
DEFAULT_EPOCHS = 5
GAMMA = 0.7        # StepLR decay factor — identical to main.py
TEST_BATCH_SIZE = 1000
SEED = 1

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_data_loaders(batch_size: int, test_batch_size: int, use_accel: bool):
    """Return (train_loader, test_loader) for MNIST."""
    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(data_root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_root, train=False, transform=transform)

    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': test_batch_size}

    if use_accel:
        accel_kwargs = {
            'num_workers': 1,
            'persistent_workers': True,
            'pin_memory': True,
            'shuffle': True,
        }
        train_kwargs.update(accel_kwargs)
        test_kwargs.update(accel_kwargs)

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def train(model, device, train_loader, optimizer, epoch):
    """Train for one epoch. Returns (avg_loss, accuracy_pct)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item() * len(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += len(data)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    print(f'  Epoch {epoch} — Train Loss: {avg_loss:.4f}, Train Acc: {accuracy:.2f}%')
    return avg_loss, accuracy


def test(model, device, test_loader):
    """Evaluate on the test set. Returns (avg_loss, accuracy_pct)."""
    model.eval()
    test_loss = 0.0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    n = len(test_loader.dataset)
    avg_loss = test_loss / n
    accuracy = 100.0 * correct / n
    print(f'           Test  Loss: {avg_loss:.4f}, Test  Acc: {accuracy:.2f}%')
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------

def run_experiment(batch_size: int, lr: float, epochs: int, device, use_accel: bool) -> dict:
    """
    Train the Net with the given hyperparameters for `epochs` epochs.

    Returns a dict with keys:
        batch_size, lr, train_losses, test_losses, train_accs, test_accs, state_dict
    """
    torch.manual_seed(SEED)

    train_loader, test_loader = get_data_loaders(batch_size, TEST_BATCH_SIZE, use_accel)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=GAMMA)

    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    for epoch in range(1, epochs + 1):
        tl, ta = train(model, device, train_loader, optimizer, epoch)
        vl, va = test(model, device, test_loader)
        scheduler.step()

        train_losses.append(tl)
        test_losses.append(vl)
        train_accs.append(ta)
        test_accs.append(va)

    return {
        'batch_size': batch_size,
        'lr': lr,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'state_dict': model.state_dict(),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_experiment_group(results: list, vary_param: str, epochs: int, save_path: str):
    """
    Save a 2x2 figure with train/test loss and train/test accuracy curves.

    vary_param: 'batch_size' or 'lr' — used to build the legend labels.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    x = range(1, epochs + 1)

    for result in results:
        label = (
            f"batch={result['batch_size']}" if vary_param == 'batch_size'
            else f"lr={result['lr']}"
        )
        axes[0, 0].plot(x, result['train_losses'], label=label)
        axes[0, 1].plot(x, result['test_losses'],  label=label)
        axes[1, 0].plot(x, result['train_accs'],   label=label)
        axes[1, 1].plot(x, result['test_accs'],    label=label)

    titles = ['Train Loss', 'Test Loss', 'Train Accuracy (%)', 'Test Accuracy (%)']
    ylabels = ['Loss', 'Loss', 'Accuracy (%)', 'Accuracy (%)']
    for ax, title, ylabel in zip(axes.flat, titles, ylabels):
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

    param_display = 'Batch Size' if vary_param == 'batch_size' else 'Learning Rate'
    fig.suptitle(f'MNIST CNN — Varying {param_display}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'Plot saved: {save_path}')


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_results_csv(all_results: list, save_path: str):
    """Write per-epoch metrics for all experiments to a CSV file."""
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['batch_size', 'lr', 'epoch',
                         'train_loss', 'test_loss', 'train_acc', 'test_acc'])
        for r in all_results:
            epochs = len(r['train_losses'])
            for epoch in range(1, epochs + 1):
                writer.writerow([
                    r['batch_size'],
                    r['lr'],
                    epoch,
                    f"{r['train_losses'][epoch - 1]:.6f}",
                    f"{r['test_losses'][epoch - 1]:.6f}",
                    f"{r['train_accs'][epoch - 1]:.4f}",
                    f"{r['test_accs'][epoch - 1]:.4f}",
                ])
    print(f'Results saved: {save_path}')


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='MNIST CNN Hyperparameter Experiments')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                        help=f'epochs per experiment (default: {DEFAULT_EPOCHS})')
    args = parser.parse_args()
    epochs = args.epochs

    # Device detection — mirrors main.py
    try:
        use_accel = torch.accelerator.is_available()
        device = torch.accelerator.current_accelerator() if use_accel else torch.device('cpu')
    except AttributeError:
        # Older PyTorch without torch.accelerator
        use_accel = torch.cuda.is_available()
        device = torch.device('cuda' if use_accel else 'cpu')

    print(f'Using device: {device}')
    print(f'Epochs per experiment: {epochs}\n')

    # ------------------------------------------------------------------
    # Experiment group 1: vary batch size (fixed lr = FIXED_LR)
    # ------------------------------------------------------------------
    print('=' * 60)
    print(f'EXPERIMENT GROUP 1: Varying batch size (lr={FIXED_LR})')
    print('=' * 60)
    batch_results = []
    for bs in BATCH_SIZES:
        print(f'\n--- batch_size={bs}, lr={FIXED_LR} ---')
        result = run_experiment(bs, FIXED_LR, epochs, device, use_accel)
        batch_results.append(result)

    # ------------------------------------------------------------------
    # Experiment group 2: vary learning rate (fixed batch = FIXED_BATCH)
    # ------------------------------------------------------------------
    print('\n' + '=' * 60)
    print(f'EXPERIMENT GROUP 2: Varying learning rate (batch={FIXED_BATCH})')
    print('=' * 60)
    lr_results = []
    for lr in LEARNING_RATES:
        print(f'\n--- batch_size={FIXED_BATCH}, lr={lr} ---')
        result = run_experiment(FIXED_BATCH, lr, epochs, device, use_accel)
        lr_results.append(result)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    print('\nGenerating plots...')
    plot_experiment_group(
        batch_results, 'batch_size', epochs,
        os.path.join(OUTPUT_DIR, 'batch_size_experiment.png')
    )
    plot_experiment_group(
        lr_results, 'lr', epochs,
        os.path.join(OUTPUT_DIR, 'lr_experiment.png')
    )

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------
    all_results = batch_results + lr_results
    save_results_csv(all_results, os.path.join(OUTPUT_DIR, 'experiment_results.csv'))

    # ------------------------------------------------------------------
    # Best model
    # ------------------------------------------------------------------
    best = max(all_results, key=lambda r: r['test_accs'][-1])
    best_path = os.path.join(OUTPUT_DIR, 'mnist_cnn_best.pt')
    torch.save(best['state_dict'], best_path)
    print(f'\nBest model: batch_size={best["batch_size"]}, lr={best["lr"]}, '
          f'test_acc={best["test_accs"][-1]:.2f}%')
    print(f'Saved to: {best_path}')

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    print(f'{"Config":<30} {"Train Acc":>10} {"Test Acc":>10}')
    print('-' * 52)
    for r in all_results:
        config = f'batch={r["batch_size"]}, lr={r["lr"]}'
        print(f'{config:<30} {r["train_accs"][-1]:>9.2f}% {r["test_accs"][-1]:>9.2f}%')


if __name__ == '__main__':
    main()
