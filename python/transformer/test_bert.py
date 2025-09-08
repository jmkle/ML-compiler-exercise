from transformers import BertTokenizer, BertModel

from torch_mlir import fx
from torch_mlir.compiler_utils import (
    OutputType,
    run_pipeline_with_repro_report,
)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')

module = fx.export_and_import( model, **encoded_input, output_type="linalg-on-tensors", func_name=model.__class__.__name__, )