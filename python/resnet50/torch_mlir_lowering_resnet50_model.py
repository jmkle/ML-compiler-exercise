from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from datasets import load_dataset
import torchvision
from torch_mlir import torchscript
from torch_mlir import fx
from torch_mlir.compiler_utils import (
    OutputType,
    run_pipeline_with_repro_report,
)

#dataset = load_dataset("huggingface/cats-image")
#image = dataset["test"]["image"][0]

#processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")

# model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
model = torchvision.models.resnet18(pretrained=True)
model.eval()

def run(f):
    #print(f"{f.__name__}")
    #print("-" * len(f.__name__))
    f()
    print()

@run
def lower_pytorch_to_linalg_on_tensors():

    # Tokenize sentences
    # inputs = processor(image, return_tensors="pt")
    """class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, input_ids):
            outputs = self.model(input_ids)
            return outputs   # <- plain tensor

    wrapped_model = Wrapper(model)"""
    
    """ep = torch.export.export(
        model,
        (torch.ones(1, 3, 224, 224),),
    )

    ep = ep.run_decompositions()"""
    
    # Export model to torch-mlir
    m = fx.export_and_import(model, torch.ones(1, 3, 224, 224), output_type=OutputType.TORCH, 
                             func_name = "resnet50_model")
    
    
    # Model in torch dialect
    mlir_str = str(m)
    mlir_ir = mlir_str.split("{-#")[0].strip()
    #print(mlir_ir)

    with open("resnet50_model_torch.mlir", "w") as f:
        f.write(mlir_str)