import os
import time
import argparse
import threading
import queue
from glob import glob
import cv2
import numpy as np
import torch
from tqdm import tqdm
import onnxruntime
from facenet_pytorch import MTCNN

# -----------------------------
# Image utils
# -----------------------------
def load_rgb_uint8(img_path: str) -> np.ndarray:
    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cv2.imread failed: {img_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.uint8)

def resize_rgb_uint8_cv2(img_rgb: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    return cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR).astype(np.uint8)

def downscale_max_side(img_rgb: np.ndarray, max_side: int):
    if max_side <= 0:
        return img_rgb, 1.0
    H, W = img_rgb.shape[:2]
    m = max(H, W)
    if m <= max_side:
        return img_rgb, 1.0
    s = float(max_side) / float(m)
    new_w = max(1, int(round(W * s)))
    new_h = max(1, int(round(H * s)))
    img2 = resize_rgb_uint8_cv2(img_rgb, new_w, new_h)
    return img2, s

def facenet_preprocess_torch(face_rgb_uint8_160: np.ndarray, device: torch.device) -> torch.Tensor:
    x = face_rgb_uint8_160.astype(np.float32)
    x = (x - 127.5) * 0.0078125
    x = np.transpose(x, (2, 0, 1))  # CHW
    x = np.expand_dims(x, 0)        # NCHW
    return torch.from_numpy(x).to(device=device, dtype=torch.float32)

# -----------------------------
# ONNX Wrapper
# -----------------------------
class GenericONNXModel(torch.nn.Module):
    def __init__(self, onnx_path: str, name: str = ""):
        super().__init__()
        self.name = name or os.path.basename(onnx_path)
        sess_options = onnxruntime.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = onnxruntime.InferenceSession(
            onnx_path, 
            sess_options=sess_options,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        # Dummy param to satisfy next(model.parameters()) check in facenet_pytorch
        self.dummy_param = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor):
        x_np = x.detach().cpu().numpy()
        if x_np.dtype != np.float32:
            x_np = x_np.astype(np.float32)
        outs = self.session.run(self.output_names, {self.input_name: x_np})
        outs_torch = [torch.from_numpy(o) for o in outs]
        if len(outs_torch) == 1:
            return outs_torch[0]
        return tuple(outs_torch)

# -----------------------------
# Alignment
# -----------------------------
def umeyama(src: np.ndarray, dst: np.ndarray, estimate_scale: bool = True) -> np.ndarray:
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)
    num = src.shape[0]
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean
    A = (dst_demean.T @ src_demean) / num
    U, S, Vt = np.linalg.svd(A)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    if estimate_scale:
        var_src = np.sum(src_demean ** 2) / num
        scale = np.sum(S) / (var_src + 1e-12)
    else:
        scale = 1.0
    t = dst_mean - scale * (R @ src_mean)
    M = np.zeros((2, 3), dtype=np.float64)
    M[:, :2] = scale * R
    M[:, 2] = t
    M[:, 2] = t
    return M.astype(np.float32)

def align_face_160(img_rgb_uint8: np.ndarray, lm5_xy: np.ndarray, image_size: int = 160) -> np.ndarray:
    ref_112 = np.array([
        [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
        [41.5493, 92.3655], [70.7299, 92.2041]
    ], dtype=np.float32)
    scale = image_size / 112.0
    ref = ref_112 * scale
    M = umeyama(lm5_xy, ref, estimate_scale=True)
    aligned = cv2.warpAffine(img_rgb_uint8, M, (image_size, image_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return aligned.astype(np.uint8)

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)

# -----------------------------
# Core Logic
# -----------------------------
def load_gallery(gallery_root, resnet, mtcnn, device, det_max_side):
    print(f"Loading Gallery from {gallery_root}...")
    gallery_embs = []
    gallery_ids = []
    
    if not os.path.exists(gallery_root):
        print("Gallery root not found!")
        return np.array([]), np.array([])
        
    for pid_name in sorted(os.listdir(gallery_root)):
        pid_path = os.path.join(gallery_root, pid_name)
        if not os.path.isdir(pid_path):
            continue
        
        img_paths = glob(os.path.join(pid_path, "**", "*.jpg"), recursive=True)
        
        for img_path in img_paths:
            try:
                img_rgb = load_rgb_uint8(img_path)
                
                img_det, ds = downscale_max_side(img_rgb, det_max_side)
                boxes, probs, points = mtcnn.detect(img_det, landmarks=True)
                
                if boxes is None or len(boxes) == 0:
                    img_det, ds = downscale_max_side(img_rgb, 640)
                    boxes, probs, points = mtcnn.detect(img_det, landmarks=True)
                
                if boxes is not None and len(boxes) > 0:
                    best = int(np.argmax(probs))
                    lm5 = points[best]
                    if ds != 1.0:
                        lm5 = lm5 / ds
                    
                    aligned = align_face_160(img_rgb, lm5)
                    x = facenet_preprocess_torch(aligned, device)
                    emb = resnet(x)[0].detach().cpu().numpy()
                    
                    gallery_embs.append(emb)
                    gallery_ids.append(pid_name)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error loading {img_path}: {repr(e)}")

    if len(gallery_embs) > 0:
        return np.stack(gallery_embs), np.array(gallery_ids)
    return np.array([]), np.array([])

def match_face(emb, gallery_embs, gallery_ids, threshold):
    if len(gallery_embs) == 0:
        return "Unknown", 0.0
    
    emb = l2_normalize(emb.reshape(1, -1))
    sims = emb @ gallery_embs.T 
    sims = sims[0]
    
    best_idx = np.argmax(sims)
    best_score = sims[best_idx]
    
    if best_score >= threshold:
        return gallery_ids[best_idx], best_score
    return "Unknown", best_score

# -----------------------------
# Heuristic Logic
# -----------------------------
def calculate_focus_score(box, frame_w, frame_h):
    # box: [x1, y1, x2, y2]
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2
    
    # 1. Area Score (Goal: Face takes up significant portion)
    # We expect face to be at least 10% of frame for interaction, up to 100%
    area_ratio = (w * h) / (frame_w * frame_h)
    score_area = min(area_ratio / 0.4, 1.0) # Cap at 40% area
    
    # 2. Center Score (Goal: Face is in middle)
    frame_cx = frame_w / 2
    frame_cy = frame_h / 2
    max_dist = np.sqrt(frame_cx**2 + frame_cy**2)
    dist = np.sqrt((cx - frame_cx)**2 + (cy - frame_cy)**2)
    score_center = 1.0 - (dist / (max_dist + 1e-6))
    
    # Weighted Sum
    # Area is dominant for "Intent" (Proximity), Center is secondary
    final_score = 0.7 * score_area + 0.3 * score_center
    return final_score

def check_permission(identity):
    # Simple ACL Mock
    if identity == "Unknown":
        return False
    return True

# -----------------------------
# Async Implementation
# -----------------------------
class AsyncVideoProcessor:
    def __init__(self, args, gallery_embs_norm, gallery_ids, mtcnn, resnet, device):
        self.args = args
        self.gallery_embs_norm = gallery_embs_norm
        self.gallery_ids = gallery_ids
        self.mtcnn = mtcnn
        self.resnet = resnet
        self.device = device
        
        self.input_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        
        self.frame_width = 0
        self.frame_height = 0
        
        # --- Temporal Logic Variables ---
        self.consecutive_count = 0
        self.last_open_id = None
        self.REQUIRED_CONSECUTIVE = 3
        # --------------------------------
        
        self.thread = threading.Thread(target=self.inference_worker)
        self.thread.daemon = True
        self.thread.start()

    def inference_worker(self):
        while not self.stop_event.is_set():
            try:
                # Get latest frame
                frame_rgb = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            # --- Inference Block ---
            try:
                final_status = "CLOSE" # Default
                reason = ""
                img_det, ds = downscale_max_side(frame_rgb, self.args.det_max_side)
                boxes, probs, points = self.mtcnn.detect(img_det, landmarks=True)
                
                results = []
                if boxes is not None:
                    # Logic: Find the "Best" face for entry
                    best_focus_score = -1.0
                    target_idx = -1
                    
                    frame_h, frame_w = frame_rgb.shape[:2]

                    for i, box in enumerate(boxes):
                        lm5 = points[i]
                        if ds != 1.0:
                            lm5 = lm5 / ds
                            box = box / ds
                        
                        # --- Logic A: Focus Score ---
                        f_score = calculate_focus_score(box, frame_w, frame_h)
                        if f_score > best_focus_score:
                            best_focus_score = f_score
                            target_idx = i

                        aligned = align_face_160(frame_rgb, lm5)
                        x = facenet_preprocess_torch(aligned, self.device)
                        emb = self.resnet(x)[0].detach().cpu().numpy()
                        
                        match_id, score = match_face(emb, self.gallery_embs_norm, self.gallery_ids, self.args.threshold_sim)
                        
                        # Save result
                        results.append({
                            'label': match_id,
                            'score': score,
                            'box': box.astype(int),
                            'focus_score': f_score
                        })
                    
                    # --- Logic B: Authorization Decision ---
                    # Only OPEN if the "Best Focus" face is recognized and allowed
                    reason = ""
                    if target_idx != -1:
                        target = results[target_idx]
                        
                        # Criteria Met?
                        is_qualified = False
                        
                        if target['focus_score'] <= 0.3:
                            reason = f"Focus Too Low ({target['focus_score']:.2f})"
                        elif target['score'] < self.args.threshold_sim:
                            reason = f"Sim Too Low ({target['score']:.2f})"
                        elif not check_permission(target['label']):
                            reason = f"No Permission ({target['label']})"
                        else:
                            is_qualified = True
                        
                        # --- Temporal Logic ---
                        if is_qualified:
                            if target['label'] == self.last_open_id:
                                self.consecutive_count += 1
                            else:
                                self.consecutive_count = 1
                                self.last_open_id = target['label']
                            
                            if self.consecutive_count >= self.REQUIRED_CONSECUTIVE:
                                final_status = "OPEN"
                                reason = f"Welcome {target['label']} (Confirmed)"
                            else:
                                final_status = "CLOSE"
                                reason = f"Verifying {target['label']}... ({self.consecutive_count}/{self.REQUIRED_CONSECUTIVE})"
                        else:
                             # Reset if current best target is not qualified
                             # Note: You might want to decay instead of hard reset, but hard reset is safer for now
                             self.consecutive_count = 0
                             self.last_open_id = None
                             # reason is already set above by failure clauses
                    else:
                        reason = "No Target Found"
                        self.consecutive_count = 0
                        self.last_open_id = None
                else:
                    self.consecutive_count = 0
                    self.last_open_id = None
                    reason = "No Faces"
                
                # Push results
                if not self.result_queue.empty():
                    _ = self.result_queue.get_nowait()
                
                # Output tuple: (detections, system_status, reason_text)
                self.result_queue.put((results, final_status, reason))
                
            except Exception as e:
                # import traceback
                # traceback.print_exc()
                pass # Silently fail detection frame to keep going

    def run(self):
        cap = cv2.VideoCapture(self.args.input_video)
        if not cap.isOpened():
            print(f"Cannot open video: {self.args.input_video}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames_one = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if self.args.loop == -1:
            repeat_count = 999999999
            infinite = True
        else:
            repeat_count = self.args.loop if self.args.loop > 0 else 1
            infinite = False
            
        total_frames_all = total_frames_one * repeat_count

        if self.args.save:
            os.makedirs("video_output/frame", exist_ok=True)
            
        out = None
        if self.args.output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.args.output, fourcc, fps, (width, height))
        
        pbar = tqdm(total=total_frames_all) if not infinite else tqdm()
        
        fps_avg = 0.0
        current_detections = [] # [(label, score, box), ...]
        current_status = "CLOSE"
        current_reason = ""
        
        loop_i = 0
        while True:
            if not infinite and loop_i >= repeat_count:
                break
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Sync reference
            t_prev = time.time()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                t_curr = time.time()
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 1. Update detections if available
                try:
                    new_res = self.result_queue.get_nowait()
                    current_detections, current_status, current_reason = new_res
                except queue.Empty:
                    pass

                # 2. Feed inference if idle
                if self.input_queue.empty():
                    self.input_queue.put(frame_rgb)
                
                # 3. Draw (Text only)
                for det in current_detections:
                    label = det['label']
                    score = det['score']
                    x1, y1, x2, y2 = det['box']
                    
                    # if score < 0.45:
                    #     continue
                    
                    if label != "Unknown":
                         # Only draw known IDs as requested ("if ID in gallery")
                         # Actually match_face returns "Unknown" if low score, but here we already check score < 0.45.
                         # If match_face returns specific ID, we assume it's "in gallery".
                         # User said: "if found ID in gallery then puttext".
                         
                         # Center of face or Top-Left
                         text_pos = (x1, max(y1 - 10, 10))
                         
                         # Outline text for visibility
                         # Add Focus Score to label for debugging
                         text = f"{label} {score:.2f} (F:{det.get('focus_score', 0.0):.2f})"
                         cv2.putText(frame, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3) # Black border
                         cv2.putText(frame, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2) # Green text

                # Draw Top Center Status
                status_color = (0, 0, 255) # Red for CLOSE
                if current_status == "OPEN":
                    status_color = (0, 255, 0) # Green for OPEN
                
                # Text size check
                label_size, baseline = cv2.getTextSize(current_status, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 5)
                top_x = (width - label_size[0]) // 2
                top_y = 60 # Margin top
                
                cv2.putText(frame, current_status, (top_x, top_y), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,0,0), 6) # Border
                cv2.putText(frame, current_status, (top_x, top_y), cv2.FONT_HERSHEY_SIMPLEX, 2.0, status_color, 5)

                # Draw Reason below status
                if current_reason:
                    r_size, _ = cv2.getTextSize(current_reason, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                    r_x = (width - r_size[0]) // 2
                    r_y = top_y + 40
                    cv2.putText(frame, current_reason, (r_x, r_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 4)
                    cv2.putText(frame, current_reason, (r_x, r_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200,200,200), 2)

                # 4. Output
                if self.args.save:
                    save_path = f"video_output/frame/frame_{pbar.n:06d}.jpg"
                    cv2.imwrite(save_path, frame)

                if out:
                    out.write(frame)
                    
                if self.args.loop != 0:
                    cv2.imshow("Video Processing V3", frame)
                    
                    # --- Global Sync Logic (Preserved) ---
                    target_time = t_prev + (1.0 / fps)
                    now = time.time()
                    
                    if now < target_time:
                        wait_ms = int((target_time - now) * 1000)
                        if wait_ms > 0:
                            if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
                                self.stop_event.set()
                                cap.release()
                                if out: out.release()
                                cv2.destroyAllWindows()
                                return
                        else:
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                self.stop_event.set() 
                                cap.release()
                                if out: out.release()
                                cv2.destroyAllWindows()
                                return
                    else:
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            self.stop_event.set()
                            cap.release()
                            if out: out.release()
                            cv2.destroyAllWindows()
                            return
                        t_prev = time.time()
                    
                    if now < target_time:
                         t_prev = target_time
                        
                pbar.update(1)
            loop_i += 1
            
        pbar.close()
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        self.stop_event.set()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video", type=str, required=True, help="Path to input mp4")
    parser.add_argument("--output", type=str, default=None, help="Path to output video")
    parser.add_argument("--threshold_sim", type=float, default=0.74)
    parser.add_argument("--det_max_side", type=int, default=320)
    parser.add_argument("--interval_frame", type=int, default=5, help="Unused in async mode (runs continuously)")
    parser.add_argument("--loop", type=int, default=0, help="Loop count for display stream (0=no display, -1=infinite)")
    parser.add_argument("--save", action="store_true", help="Save frames to video_output/frame")
    parser.add_argument("--device", type=str, default="auto")
    
    args = parser.parse_args()
    
    if args.device == "cuda":
        device = torch.device("cuda:0")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    print(f"Using device: {device}")
    
    # Init Models
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.7, 0.8, 0.9], factor=0.709, post_process=False,
        keep_all=True, device=device
    )
    
    models_dir = os.path.join(os.path.dirname(__file__), "onnx_models")
    pnet_path = os.path.join(models_dir, "mtcnn_pnet.onnx")
    rnet_path = os.path.join(models_dir, "mtcnn_rnet.onnx")
    onet_path = os.path.join(models_dir, "mtcnn_onet.onnx")
    resnet_path = os.path.join(models_dir, "facenet_resnet.onnx")
    
    mtcnn.pnet = GenericONNXModel(pnet_path)
    mtcnn.rnet = GenericONNXModel(rnet_path)
    mtcnn.onet = GenericONNXModel(onet_path)
    resnet = GenericONNXModel(resnet_path)
    
    # Load Gallery
    gallery_root = "filter_gallery"
    gallery_embs, gallery_ids = load_gallery(gallery_root, resnet, mtcnn, device, args.det_max_side)
    if len(gallery_embs) > 0:
        gallery_embs_norm = l2_normalize(gallery_embs)
    else:
        gallery_embs_norm = np.array([])
        
    print(f"Loaded {len(gallery_embs)} identities from gallery.")
    
    # Run Async Processor
    processor = AsyncVideoProcessor(args, gallery_embs_norm, gallery_ids, mtcnn, resnet, device)
    processor.run()

if __name__ == "__main__":
    main()
