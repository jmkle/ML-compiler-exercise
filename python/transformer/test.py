from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Model
import torch

from torch_mlir import fx
from torch_mlir.compiler_utils import (
    OutputType,
    run_pipeline_with_repro_report,
)

torch.manual_seed(41)

test_modelname = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(test_modelname)
prompt = "What is nature of our existence?"

model = GPT2LMHeadModel.from_pretrained(
    test_modelname,
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False,
    torchscript=True,
    attn_implementation="eager",
)

model.to("cpu")
model.eval()

def run(f):
    #print(f"{f.__name__}")
    #print("-" * len(f.__name__))
    f()
    print()

@run
def lower_pytorch_to_linalg_on_tensors():

    # Tokenize sentences
    encoding = tokenizer(prompt, return_tensors="pt")

    
    # Export model to torch-mlir
    m = fx.export_and_import( model, **encoding, output_type=OutputType.LINALG_ON_TENSORS, func_name=model.__class__.__name__, )

    
    # Model in torch dialect
    mlir_str = str(m)
    mlir_ir = mlir_str.split("{-#")[0].strip()
    #print(mlir_ir)

    with open("transformer_model_linalgOnTensor.mlir", "w") as f:
        f.write(mlir_str)