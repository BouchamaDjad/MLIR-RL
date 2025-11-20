# export_bert_to_onnx.py
from transformers import AutoTokenizer, AutoModel, BartForConditionalGeneration, BartTokenizer
import torch
from torch.export import export

output_file = "T5"
model = "T5"

input_names=None
output_names=None

# Each block should define `example_inputs` and `model` and optionally `input_names` and `output_names`
if model == "bert":
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()

    # dummy input
    inputs = tokenizer("Hello from MLIR", return_tensors="pt", padding="max_length", max_length=16, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    # token_type_ids = inputs.get("token_type_ids", torch.zeros_like(input_ids))

    example_inputs = (input_ids, attention_mask)
    input_names=["input_ids","attention_mask"]
    output_names=["last_hidden_state"]

elif model == "bart":
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    # Prepare dummy batch
    src_text = ["The cat sat on the mat."]
    # tgt_text = ["Le chat s'est assis sur le tapis."]

    src_enc = tokenizer(src_text, return_tensors="pt", padding="max_length", truncation=True, max_length=16)
    # tgt_enc = tokenizer(tgt_text, return_tensors="pt", padding="max_length", truncation=True, max_length=16)

    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

    example_inputs = (src_enc["input_ids"], src_enc["attention_mask"])
    input_names=["input_ids","attention_mask"]
    output_names=["last_hidden_state"]

elif model == "graph":
    from torch_geometric.nn import GraphSAGE

    # Example usage
    from torch_geometric.datasets import Planetoid
    dataset = Planetoid(root="data", name="Pubmed")
    data = dataset[0]

    model = GraphSAGE(
        in_channels=dataset.num_features,
        hidden_channels=64,
        num_layers=2,
        out_channels=dataset.num_classes,
        dropout=0.5,
    )
    example_inputs = (data.x, data.edge_index)

elif model == "gpt":
    from transformers import GPT2Tokenizer, GPT2Model
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    model = GPT2Model.from_pretrained('distilgpt2')
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors="pt", max_length=16, truncation=True)

    # input_names=["input_ids","attention_mask"]
    output_names=["last_hidden_state"]

    # example_inputs = (encoded_input['input_ids'], encoded_input["attention_mask"])
    example_inputs = dict(encoded_input)

elif model == "T5":
    from transformers import T5Tokenizer, T5Model

    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5Model.from_pretrained("t5-small")

    input_ids = tokenizer("Studies have been shown that owning a dog is good for you", padding="max_length", return_tensors="pt").input_ids  # Batch size 1
    decoder_input_ids = tokenizer("Studies show that", padding="max_length" ,return_tensors="pt").input_ids  # Batch size 1

    example_inputs = {
        "input_ids": input_ids,
        "decoder_input_ids": decoder_input_ids
    }
    # input_names = ["input_ids", "decoder_input_ids"]
    output_names = ["last_hidden_state"]
    

torch.onnx.export(
    model,
    example_inputs,
    f"{output_file}.onnx",
    opset_version=17,                           # try 13 or 17; if you hit shape/compat issues try higher opset
    input_names=input_names,
    output_names=output_names,
    # dynamic_axes={
    #     "input_ids": {0: "batch", 1: "seq"},
    #     "attention_mask": {0: "batch", 1: "seq"},
    #     # "token_type_ids": {0: "batch", 1: "seq"},
    #     "last_hidden_state": {0: "batch", 1: "seq"},
    # },
    do_constant_folding=True,
    # dynamo=True
)


import onnx

onnx.checker.check_model(onnx.load(f"{output_file}.onnx"))

print('OK')


from os import system

system(
f"python -m onnxruntime.tools.symbolic_shape_infer \
--input {output_file}.onnx \
--output {output_file}_inferred.onnx \
--auto_merge"
)

# from the torch-mlir repo build, or if torch-mlir is installed as a package:
system(
    f"python -m torch_mlir.tools.import_onnx {output_file}_inferred.onnx -o {output_file}_torch.mlir --opset-version 17"
)

# converts onnx.* torch.operator to actual torch.* ops where legalizers exist
# torch-mlir-opt bert_torch.mlir --convert-torch-onnx-to-torch -o bert_torch_legalized.mlir

# convert torch ops to linalg where possible
# torch-mlir-opt bert_torch_legalized.mlir --convert-torch-to-linalg -o bert_linalg.mlir (do not run)

# torch-mlir-opt bert_torch_legalized.mlir \
#   --torch-decompose-complex-ops \
#   --convert-torch-to-linalg \
#   --torch-backend-to-linalg-on-tensors-backend-pipeline \
#   -o bert_linalg.mlir

system(
    f"torch-mlir-opt {output_file}_torch.mlir \
--convert-torch-onnx-to-torch \
--torch-decompose-complex-ops \
--convert-torch-to-linalg \
--torch-backend-to-linalg-on-tensors-backend-pipeline \
-o {output_file}_linalg.mlir"
)

print(f"\033[92m file succeesfully created in \033[1m{output_file}_linalg.mlir\033[0m")