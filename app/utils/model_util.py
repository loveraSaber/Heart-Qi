import torch 
from src.models.single_modal_backbone import E2E

def load_ckpt(config, ckpt, device):
   
    model = E2E(config.backbone, config.scale_type).to(device)
    if ckpt.endswith('pth'):
        model.load_state_dict(
            torch.load(ckpt, map_location=device)
        )
    else:
        ckpt = torch.load(ckpt, map_location=device, weights_only=False)
        state_dict = {}
        for key, value in ckpt['state_dict'].items():
            if key[:6] == 'model.':
                state_dict[key[6:]] = value
        model.load_state_dict(state_dict, strict=True)
    # 选择验证模式
    model.eval()
    
    return model
