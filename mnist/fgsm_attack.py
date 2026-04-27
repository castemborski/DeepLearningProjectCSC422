"""Goal 2: FGSM Adversarial Attack on the Best MNIST CNN

Loads the best model from Goal 1, evaluates it under FGSM attacks at
several epsilon values, and saves two output figures:
  fgsm_accuracy_vs_epsilon.png  — accuracy curve across epsilon values
  fgsm_examples.png             — original vs adversarial image grids
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import Net  # noqa: E402

EPSILONS = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
VIS_EPSILONS = [0.05, 0.15, 0.3]   # epsilons used in the example-grid figure
N_EXAMPLES = 5                      # examples per epsilon row

# Clamp range in normalized pixel space (preserves valid MNIST pixel values)
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081
NORM_MIN = (0.0 - MNIST_MEAN) / MNIST_STD   # ≈ -0.4242
NORM_MAX = (1.0 - MNIST_MEAN) / MNIST_STD   # ≈  2.8215

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mnist_cnn_best.pt')
DATA_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
OUT_DIR    = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Attack
# ---------------------------------------------------------------------------

def fgsm_attack(image: torch.Tensor, epsilon: float, data_grad: torch.Tensor) -> torch.Tensor:
    """Fast Gradient Sign Method perturbation (clamped to valid normalized range)."""
    perturbed = image + epsilon * data_grad.sign()
    return torch.clamp(perturbed, NORM_MIN, NORM_MAX)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_under_attack(model, device, test_loader, epsilon):
    """
    Run FGSM at the given epsilon over the full test set.

    Returns:
        accuracy  (float)   — fraction of correctly classified adversarial examples
        examples  (list)    — up to N_EXAMPLES tuples:
                              (true_label, orig_img_np, orig_pred, adv_img_np, adv_pred)
                              collected from cases where the original prediction was correct
    """
    model.eval()
    correct = 0
    total = 0
    examples = []

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad_(True)

        output = model(data)
        orig_pred = output.argmax(dim=1)

        loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()

        if epsilon == 0:
            perturbed = data.detach()
        else:
            perturbed = fgsm_attack(data, epsilon, data.grad.data)

        with torch.no_grad():
            adv_output = model(perturbed)
            adv_pred = adv_output.argmax(dim=1)

        correct += adv_pred.eq(target).sum().item()
        total += len(target)

        # Collect visualisation examples from correctly-classified originals
        for i in range(len(target)):
            if len(examples) < N_EXAMPLES and orig_pred[i].item() == target[i].item():
                examples.append((
                    target[i].item(),
                    data[i].detach().cpu().squeeze().numpy(),
                    orig_pred[i].item(),
                    perturbed[i].detach().cpu().squeeze().numpy(),
                    adv_pred[i].item(),
                ))

    accuracy = correct / total
    return accuracy, examples


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_accuracy_curve(epsilons, accuracies, save_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epsilons, [a * 100 for a in accuracies],
            'o-', color='steelblue', linewidth=2, markersize=8)
    ax.set_xlabel('Epsilon (attack budget)', fontsize=13)
    ax.set_ylabel('Test Accuracy (%)', fontsize=13)
    ax.set_title('FGSM Attack: Model Accuracy vs. Epsilon', fontsize=14)
    ax.set_xticks(epsilons)
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3)
    for eps, acc in zip(epsilons, accuracies):
        ax.annotate(f'{acc * 100:.1f}%',
                    xy=(eps, acc * 100), xytext=(0, 9),
                    textcoords='offset points', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'Saved: {save_path}')


def plot_examples(all_examples, vis_epsilons, save_path):
    n_rows = len(vis_epsilons) * 2   # original + adversarial row per epsilon
    fig, axes = plt.subplots(n_rows, N_EXAMPLES, figsize=(N_EXAMPLES * 2.2, n_rows * 2.4))
    fig.suptitle('Original vs. Adversarial Examples (FGSM)', fontsize=14, y=1.01)

    for pair_idx, eps in enumerate(vis_epsilons):
        examples = all_examples[eps]
        r_orig = pair_idx * 2
        r_adv  = pair_idx * 2 + 1

        for col in range(N_EXAMPLES):
            true_label, orig_img, orig_pred, adv_img, adv_pred = examples[col]

            # Original image row
            ax = axes[r_orig][col]
            ax.imshow(orig_img, cmap='gray', vmin=orig_img.min(), vmax=orig_img.max())
            ax.axis('off')
            color = 'green' if orig_pred == true_label else 'red'
            ax.set_title(f'Pred: {orig_pred}', fontsize=9, color=color)
            if col == 0:
                ax.set_ylabel(f'ε={eps}\nOriginal', fontsize=9, labelpad=40)

            # Adversarial image row
            ax = axes[r_adv][col]
            ax.imshow(adv_img, cmap='gray', vmin=adv_img.min(), vmax=adv_img.max())
            ax.axis('off')
            color = 'green' if adv_pred == true_label else 'red'
            ax.set_title(f'Pred: {adv_pred}', fontsize=9, color=color)
            if col == 0:
                ax.set_ylabel('Adversarial', fontsize=9, labelpad=40)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {save_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device('cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = Net().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print(f'Loaded model: {MODEL_PATH}\n')

    accuracies  = []
    all_examples = {}

    print(f'{"Epsilon":<10} {"Correct":>8} {"Total":>8} {"Accuracy":>10}')
    print('-' * 40)
    for eps in EPSILONS:
        acc, examples = evaluate_under_attack(model, device, test_loader, eps)
        accuracies.append(acc)
        all_examples[eps] = examples
        n_correct = int(round(acc * len(test_dataset)))
        print(f'{eps:<10.2f} {n_correct:>8} {len(test_dataset):>8} {acc * 100:>9.2f}%')

    # Accuracy vs epsilon curve
    plot_accuracy_curve(
        EPSILONS, accuracies,
        os.path.join(OUT_DIR, 'fgsm_accuracy_vs_epsilon.png'),
    )

    # Original vs adversarial example grid
    plot_examples(
        all_examples, VIS_EPSILONS,
        os.path.join(OUT_DIR, 'fgsm_examples.png'),
    )

    print('\nDone.')


if __name__ == '__main__':
    main()
