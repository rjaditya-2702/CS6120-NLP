from nnsight import LanguageModel
import numpy as np

import matplotlib.pyplot as plt

def plot_head0_heatmap(sent1, sent2, model_folder_path):
    """
    Given two sentences and the path to the model, this function will feed each sentence to the model.
    After getting the final logits, a heat maps of embeddings of each text is plotted along with 
    heat map of absolute difference between the two embedding matrices
    (heat maps size: seq_length x 756)
    """

    model = LanguageModel(model_folder_path, device_map='auto')

    # Attention head 0's activations for text1
    with model.trace(sent1) as tracer:
        h0 = model.transformer.h[0].output[0].save()
    h0 = h0.cpu().detach().numpy()

    # Attention head 0's activations for text2
    with model.trace(sent2) as tracer:
        h1 = model.transformer.h[0].output[0].save()
    h1 = h1.cpu().detach().numpy()

    h0 = h0[0]
    h1 = h1[0]

    # compute absolute diff
    hDiff = np.abs(h0 - h1)

    vmin = min(h0.min(), h1.min())
    vmax = max(h0.max(), h1.max())

    # plot heat maps.
    fig, axes = plt.subplots(3, 1, figsize=(20, 15), constrained_layout=True)

    im0 = axes[0].imshow(h0, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_title(sent1)
    axes[0].set_xlabel('Activation')
    axes[0].set_ylabel('Words')

    im1 = axes[1].imshow(h1, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title(sent2)
    axes[1].set_xlabel('Activation')
    axes[1].set_ylabel('Words')

    im2 = axes[2].imshow(hDiff, aspect='auto', cmap='Greys')
    axes[2].set_title("Absolute difference in activations")
    axes[2].set_xlabel('Activation')
    axes[2].set_ylabel('Words')

    # Add a shared colorbar
    cbar = fig.colorbar(im0, ax=axes[0], orientation='vertical', fraction=0.02, pad=0.02)
    cbar = fig.colorbar(im1, ax=axes[1], orientation='vertical', fraction=0.02, pad=0.02)
    cbar = fig.colorbar(im2, ax=axes[2], orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label('Value')