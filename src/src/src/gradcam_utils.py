"""
Minimal Grad-CAM implementation (no extra heavy dependency needed beyond torch/matplotlib).
Grad-CAM shows which regions of an image most influenced the model's prediction —
this is what makes a classifier demo look genuinely interpretable rather than a black box.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor):
        """Returns (heatmap as HxW numpy array in [0,1], predicted class index)."""
        self.model.zero_grad()
        output = self.model(input_tensor)
        pred_class = output.argmax(dim=1).item()

        score = output[0, pred_class]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, pred_class


def overlay_heatmap(img_tensor, heatmap, mean, std, alpha=0.4):
    """
    img_tensor: normalized CHW tensor (as fed to the model)
    heatmap: HxW numpy array in [0,1] from GradCAM.generate()
    Returns an RGB uint8 numpy array (HxWx3) ready for imshow/saving.
    """
    img = img_tensor.clone().cpu().numpy().transpose(1, 2, 0)
    img = img * np.array(std) + np.array(mean)
    img = np.clip(img, 0, 1)

    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0

    overlay = (1 - alpha) * img + alpha * heatmap_color
    overlay = np.clip(overlay, 0, 1)
    return overlay
