from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

from torch_mlir import fx
from torch_mlir.compiler_utils import (
    OutputType,
    run_pipeline_with_repro_report,
)

torch.manual_seed(41)

sentences = ["This is an example sentence", "Each sentence is converted"]

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

def run(f):
    #print(f"{f.__name__}")
    #print("-" * len(f.__name__))
    f()
    print()

"""
@run
def lower_pytorch_to_torch_fx():
    from torch.fx import symbolic_trace
    # Symbolic tracing frontend - captures the semantics of the module
    symbolic_traced: torch.fx.GraphModule = symbolic_trace(NeuralNetwork())

    # High-level intermediate representation (IR) - Graph representation
    print(symbolic_traced.graph)


@run
def lower_pytorch_to_raw_output():
    # Export model to torch-mlir
    m = fx.export_and_import(ConvolutionalNetwork(), torch.randn(1, 1, 28, 28), output_type=OutputType.RAW,
                             func_name = "cnn_model")

    # Model in torch dialect
    mlir_str = str(m)
    mlir_ir = mlir_str.split("{-#")[0].strip()
    print(mlir_ir)

    with open("cnn_model_raw.mlir", "w") as f:
        f.write(mlir_str)

@run
def lower_pytorch_to_torch_mlir():
    # Export model to torch-mlir
    m = fx.export_and_import(ConvolutionalNetwork(), torch.randn(1, 1, 28, 28), output_type=OutputType.TORCH,
                             func_name = "cnn_model")

    # Model in torch dialect
    mlir_str = str(m)
    mlir_ir = mlir_str.split("{-#")[0].strip()
    print(mlir_ir)

    with open("cnn_model_torch.mlir", "w") as f:
        f.write(mlir_str)
"""

@run
def lower_pytorch_to_linalg_on_tensors():

    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, input_ids, attention_mask):
            outputs = self.model(input_ids, attention_mask=attention_mask)
            return outputs.last_hidden_state   # <- plain tensor

    wrapped_model = Wrapper(model)

    ep = torch.export.export(
        wrapped_model,
        (
            encoded_input["input_ids"],
            encoded_input["attention_mask"],
        )
    )

    ep = ep.run_decompositions()
    

    # Export model to torch-mlir
    m = fx.export_and_import(ep, output_type=OutputType.LINALG_ON_TENSORS, 
                             func_name = "transformer_model")
    
    # Model in torch dialect
    mlir_str = str(m)
    mlir_ir = mlir_str.split("{-#")[0].strip()
    #print(mlir_ir)

    with open("transformer_model_linalg.mlir", "w") as f:
        f.write(mlir_str)