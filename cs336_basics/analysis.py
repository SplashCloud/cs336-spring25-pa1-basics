import matplotlib.pyplot as plt
import os
import numpy as np

def plot_loss_curve(loss_file: str = "loss.txt", output_file: str = None, window_size: int = None):

    losses = []
    
    if not os.path.exists(loss_file):
        raise FileNotFoundError(f"Loss file not found: {loss_file}")
    
    with open(loss_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('loss='):
                try:
                    loss_value = float(line.split('=')[1])
                    losses.append(loss_value)
                except ValueError:
                    continue
    
    if len(losses) == 0:
        raise ValueError(f"No valid loss values found in {loss_file}")
    
    iterations = np.arange(1, len(losses) + 1)
    
    plt.figure(figsize=(12, 6))
    
    if window_size is not None and window_size > 1:
        smoothed_losses = []
        for i in range(len(losses)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(losses), i + window_size // 2 + 1)
            smoothed_losses.append(np.mean(losses[start_idx:end_idx]))
        
        plt.plot(iterations, losses, alpha=0.3, label='Raw Loss', linewidth=0.5)
        plt.plot(iterations, smoothed_losses, label=f'Smoothed Loss (window={window_size})', linewidth=2)
    else:
        plt.plot(iterations, losses, label='Loss', linewidth=1)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Loss curve saved to {output_file}")
    else:
        plt.show()
    
    plt.close()


def main():
    plot_loss_curve("loss.txt", output_file="loss.png", window_size=100)


main()