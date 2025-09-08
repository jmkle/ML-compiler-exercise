from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from torchvision import transforms
from PIL import Image
import requests
import torch.utils._pytree as pytree
from torch_mlir_e2e_test.configs.utils import (
    recursively_convert_to_numpy,
)

DEFAULT_IMAGE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg"
)

def load_and_preprocess_image(url: str = DEFAULT_IMAGE_URL):
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36"
    }
    img = Image.open(requests.get(url, headers=headers, stream=True).raw).convert("RGB")
    # preprocessing pipeline
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img_preprocessed = preprocess(img)
    return torch.unsqueeze(img_preprocessed, 0)

model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50").eval()
params = {
    **dict(model.named_buffers(remove_duplicate=False)),
}
params_flat, params_spec = pytree.tree_flatten(params)
params_flat = list(params_flat)
img = load_and_preprocess_image(DEFAULT_IMAGE_URL)
with torch.no_grad():
    numpy_inputs = recursively_convert_to_numpy(params_flat)
    
#print(img["pixel_values"].shape)
print(numpy_inputs[1])
print(numpy_inputs[2])


#import numpy as np

#normalized = [np.atleast_1d(arr) for arr in numpy_inputs]

"""
with open("weights.h", "w") as f:
    for i, arr in enumerate(normalized):
        arr = arr.flatten()  # ensure 1D
        f.write(f"float a{i}[] = {{")
        f.write(", ".join(f"{x}f" for x in arr.tolist()))
        f.write("};\n")
    
    f.write("\nvoid *x[] = {")
    f.write(", ".join(f"a{i}" for i in range(len(normalized))))
    f.write("};\n")
"""