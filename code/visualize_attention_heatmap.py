from nnsight import LanguageModel

import matplotlib.pyplot as plt

def plot_head0_heatmap(sent1, sent2, model_folder_path):

    model = LanguageModel(model_folder_path, device_map='auto')

    with model.trace(sent1) as tracer:
        h0 = model.transformer.h[0].output[0].save()
    h0 = h0.cpu().detach().numpy()

    with model.trace(sent2) as tracer:
        h1 = model.transformer.h[0].output[0].save()
    h1 = h1.cpu().detach().numpy()

    h0 = h0[0]
    h1 = h1[0]
    hDiff = h0 - h1
    vmin = min(h0.min(), h1.min())
    vmax = max(h0.max(), h1.max())

    fig, axes = plt.subplots(3, 1, figsize=(20, 15), constrained_layout=True)

    im0 = axes[0].imshow(h0, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_title(sent1)
    axes[0].set_xlabel('Activation')
    axes[0].set_ylabel('Words')

    im1 = axes[1].imshow(h1, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title(sent2)
    axes[1].set_xlabel('Activation')
    axes[1].set_ylabel('Words')

    im2 = axes[2].imshow(hDiff, aspect='auto', cmap='viridis')
    axes[2].set_title("Absolute difference in activations")
    axes[2].set_xlabel('Activation')
    axes[2].set_ylabel('Words')

    # Add a shared colorbar
    cbar = fig.colorbar(im0, ax=axes[0], orientation='vertical', fraction=0.02, pad=0.02)
    cbar = fig.colorbar(im1, ax=axes[1], orientation='vertical', fraction=0.02, pad=0.02)
    cbar = fig.colorbar(im2, ax=axes[2], orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label('Value')