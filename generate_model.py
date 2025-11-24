import onnx
from onnx import helper, TensorProto

# Create input tensor
input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None])

# Create output tensor
output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None])

# Create a constant multiplier tensor
scale_initializer = helper.make_tensor(
    name="scale",
    data_type=TensorProto.FLOAT,
    dims=[1],
    vals=[2.0],  # multiply input by 2
)

# Create Mul node (output = input * scale)
node = helper.make_node(
    "Mul",
    inputs=["input", "scale"],
    outputs=["output"],
)

# Create the graph
graph_def = helper.make_graph(
    [node],
    "simple-mul-model",
    [input_tensor],
    [output_tensor],
    [scale_initializer],
)

# Create the model
model_def = helper.make_model(graph_def, producer_name="tiny-onnx-gen")

# Save as model.onnx
onnx.save(model_def, "model.onnx")

print("model.onnx generated successfully!")
