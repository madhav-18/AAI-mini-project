

import torch
from torchvision.transforms.functional import to_tensor, to_pil_image

from model import Generator


torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def load_image(img, x32=False):

    if x32:
        def to_32s(x):
            return 256 if x < 256 else x - x % 32
        w, h = img.size
        img = img.resize((to_32s(w), to_32s(h)))

    return img

def load_basic_required_data() -> tuple:
    checkpoint = './weights/paprika.pt'
    device = 'cpu'

    return checkpoint, device


def test(input_image):

    checkpoint, device = load_basic_required_data()
    
    net = Generator()
    net.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    net.to(device).eval()

    image = load_image(input_image)

    with torch.no_grad():
        image = to_tensor(image).unsqueeze(0) * 2 - 1
        out = net(image.to(device), False).cpu()
        out = out.squeeze(0).clip(-1, 1) * 0.5 + 0.5
        out = to_pil_image(out)

    return out