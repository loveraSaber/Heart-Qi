import torch 
import sys
import numpy.core
import numpy.core.multiarray
from src.models.single_modal_backbone import E2E

class NumpyCorePatch:
    def __enter__(self):
        self.had_core = 'numpy._core' in sys.modules
        self.had_multiarray = 'numpy._core.multiarray' in sys.modules
        sys.modules['numpy._core'] = numpy.core
        sys.modules['numpy._core.multiarray'] = numpy.core.multiarray
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.had_core:
            sys.modules.pop('numpy._core', None)
        if not self.had_multiarray:
            sys.modules.pop('numpy._core.multiarray', None)

def load_ckpt(config, ckpt, device):
   
    model = E2E(config.backbone, config.scale_type).to(device)
    with NumpyCorePatch():
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
