import torch
import coremltools as ct
from facenet_pytorch import InceptionResnetV1
import numpy as np

def convert_torch_to_coreml(out_path):
    print("Loading PyTorch InceptionResnetV1 (vggface2)...")
    # Load model exactly as in the export script
    model = InceptionResnetV1(pretrained="vggface2").eval()
    
    # Create dummy input for tracing
    example_input = torch.rand(1, 3, 160, 160)
    
    # Trace the model
    traced_model = torch.jit.trace(model, example_input)
    
    # Define Image Input with Normalization
    # Formula: output = (input + bias) * scale
    # We need: (x - 127.5) / 128.0
    # bias = -127.5
    # scale = 1/128.0
    
    scale = 1/128.0
    bias = [-127.5, -127.5, -127.5]
    
    image_input = ct.ImageType(
        name="input",
        shape=example_input.shape,
        scale=scale,
        bias=bias,
        color_layout='RGB'
    )
    
    print("Converting to CoreML...")
    mlmodel = ct.convert(
        traced_model,
        inputs=[image_input],
        outputs=[ct.TensorType(name="embedding")], # Rename output to embedding
        minimum_deployment_target=ct.target.iOS15,
        compute_precision=ct.precision.FLOAT16
    )
    
    # Metadata
    mlmodel.author = "Antigravity Adapter"
    mlmodel.short_description = "FaceNet PyTorch -> CoreML with (x-127.5)/128 normalization"
    
    print(f"Saving to {out_path}...")
    mlmodel.save(out_path)
    print("Success!")

if __name__ == "__main__":
    convert_torch_to_coreml("onnx_models/FaceNet.mlpackage")
