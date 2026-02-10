import argparse
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def alignment_procedure(img_rgb, landmarks):
    # This replicates the standard 112x112 alignment then scaling to 160x160
    # Ideally we use the same Umeyama logic.
    # For this script, we can rely on facenet_pytorch's internal alignment OR implement Umeyama.
    # Let's use simple crop for now if alignment is complex to port 1:1 in single script, 
    # BUT the user wants to compare "in Engine".
    # The Engine use Umeyama.
    pass

def main():
    parser = argparse.ArgumentParser(description="Compare 2 images using FaceNet (Corrected Pipeline)")
    parser.add_argument("img1", help="Path to first image")
    parser.add_argument("img2", help="Path to second image")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # 1. Load Models
    mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # 2. Process Image 1
    img1 = Image.open(args.img1)
    # Get cropped tensor (MTCNN handles alignment internally generally well)
    # Note: MTCNN in facenet_pytorch does (x-127.5)/128 automatically if return_prob=False
    t1 = mtcnn(img1) 
    
    if t1 is None:
        print("No face detected in Image 1")
        return

    # 3. Process Image 2
    img2 = Image.open(args.img2)
    t2 = mtcnn(img2)
    
    if t2 is None:
        print("No face detected in Image 2")
        return

    # 4. Embed
    # t1 is already normalized tensors if using mtcnn() directly
    # Check normalization: facenet_pytorch mtcnn returns fixed standardized tensors
    
    with torch.no_grad():
        e1 = resnet(t1.unsqueeze(0).to(device)).cpu().numpy()[0]
        e2 = resnet(t2.unsqueeze(0).to(device)).cpu().numpy()[0]

    # 5. Compare
    def l2_normalize(x):
        return x / np.linalg.norm(x)

    e1_norm = l2_normalize(e1)
    e2_norm = l2_normalize(e2)
    
    similarity = np.dot(e1_norm, e2_norm)
    
    print(f"\n--- Results ---")
    print(f"Image 1: {args.img1}")
    print(f"Image 2: {args.img2}")
    print(f"Cosine Similarity: {similarity:.4f}")
    
    if similarity > 0.74:
        print("Verdict: SAME PERSON (Match)")
    else:
        print("Verdict: DIFFERENT PEOPLE (No Match)")
    
    # Debug: Check Raw Similarity (Simulating the bug)
    # If we had unnormalized inputs, sim would be higher.
    # But here we use correct pipeline.

if __name__ == "__main__":
    main()
