import sys
import os
import torch
import numpy as np
from glob import glob
from facenet_pytorch import MTCNN

# Add parent dir to path to import video_processing_v3
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from video_processing_v3 import (
        load_rgb_uint8, 
        downscale_max_side, 
        align_face_160, 
        facenet_preprocess_torch, 
        GenericONNXModel,
        load_gallery,
        match_face,
        l2_normalize
    )
except ImportError:
    print("Error: Could not import from video_processing_v3.py. Make sure you run this from Edge_case folder and video_processing_v3.py is in the parent folder.")
    sys.exit(1)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Setup Models (Same as video_processing_v3.py)
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.7, 0.8, 0.9], factor=0.709, post_process=False,
        keep_all=True, device=device
    )
    
    # Paths relative to script (in Edge_case/) -> models in ../onnx_models
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    models_dir = os.path.join(parent_dir, "onnx_models")
    
    pnet_path = os.path.join(models_dir, "mtcnn_pnet.onnx")
    rnet_path = os.path.join(models_dir, "mtcnn_rnet.onnx")
    onet_path = os.path.join(models_dir, "mtcnn_onet.onnx")
    resnet_path = os.path.join(models_dir, "facenet_resnet.onnx")
    
    mtcnn.pnet = GenericONNXModel(pnet_path)
    mtcnn.rnet = GenericONNXModel(rnet_path)
    mtcnn.onet = GenericONNXModel(onet_path)
    resnet = GenericONNXModel(resnet_path)
    
    # 2. Load Gallery
    # User requested to use Edge_case/gallery
    gallery_root = os.path.join(base_dir, "gallery")
    gallery_embs, gallery_ids = load_gallery(gallery_root, resnet, mtcnn, device, 320)
    
    if len(gallery_embs) > 0:
        gallery_embs_norm = l2_normalize(gallery_embs)
    else:
        print("Gallery empty!")
        return

    print(f"Loaded {len(gallery_embs)} identities.")

    # 3. Process Valid Images
    valid_dir = os.path.join(base_dir, "valid")
    valid_images = glob(os.path.join(valid_dir, "**", "*.jpg"), recursive=True)
    valid_images += glob(os.path.join(valid_dir, "**", "*.png"), recursive=True)
    
    print(f"Found {len(valid_images)} images in {valid_dir}")
    
    for img_path in valid_images:
        print(f"\nProcessing: {os.path.basename(img_path)}")
        try:
            img_rgb = load_rgb_uint8(img_path)
            
            # Detect
            img_det, ds = downscale_max_side(img_rgb, 640)
            boxes, probs, points = mtcnn.detect(img_det, landmarks=True)
            
            if boxes is None:
                print("  No face detected.")
                continue
                
            # Logic: Find best matching face in the image
            max_sim = -1.0
            best_id = "Unknown"
            
            for i, box in enumerate(boxes):
                lm5 = points[i]
                if ds != 1.0:
                    lm5 = lm5 / ds
                
                # Align
                aligned = align_face_160(img_rgb, lm5)
                
                # Embed
                x = facenet_preprocess_torch(aligned, device)
                emb = resnet(x)[0].detach().cpu().numpy()
                
                # Match
                # match_face normalizes emb inside
                # But we want raw max score
                emb = l2_normalize(emb.reshape(1, -1))
                sims = emb @ gallery_embs_norm.T
                sims = sims[0]
                
                curr_best_idx = np.argmax(sims)
                curr_best_score = sims[curr_best_idx]
                curr_best_id = gallery_ids[curr_best_idx]
                
                if curr_best_score > max_sim:
                    max_sim = curr_best_score
                    best_id = curr_best_id
            
            print(f"  Max Similarity: {max_sim:.4f} (Matched with {best_id})")
            
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    main()
