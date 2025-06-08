from typing import Optional
import torch
import torch.nn.functional as F


class SimpleGradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_cam(self, input_tensor: torch.Tensor, target_index: Optional[int] = None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if target_index is None:
            target_index = output.argmax(dim=1).item()

        loss = output[:, target_index]
        loss.backward()

        pooled_gradients = self.gradients.mean(dim=[0, 2, 3])
        activations = self.activations.squeeze(0)
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]

        heatmap = activations.mean(dim=0).cpu()
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)
        return heatmap.numpy()

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

