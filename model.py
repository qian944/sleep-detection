import torch
from model_cbam import ResNet18_CBAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import RegressionTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def load_model_from_hf():
    model = ResNet18_CBAM(pretrained=False)
    checkpoint = torch.hub.load_state_dict_from_url(
        'https://huggingface.co/qianxunliu/sleep-detection/resolve/main/model.pt',
        map_location=torch.device('cpu'),
        file_name='model.pt'
    )
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def get_cam(model, input_tensor):
    target_layers = [model.features[-4]]  # 最后一个 CBAM 前的 layer4 输出
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    targets = [RegressionTarget(0)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    return grayscale_cam
