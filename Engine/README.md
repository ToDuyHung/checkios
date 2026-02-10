# Face Recognition Engine - Logic & Differences

## Overview
This Engine module implements the Face Recognition pipeline on iOS/macOS using Swift. It aims to replicate the logic of our Python verification pipeline (`video_processing_v3.py`) but adapts it to the Apple ecosystem (Vision framework + CoreML).

## Key Implementation Details

### 1. Face Detection (Vision vs MTCNN)
- **Python**: Uses `MTCNN` which returns 5 landmarks in **Viewer's Perspective** (Point 0 is Left side of image).
- **Swift**: Uses Apple `Vision` framework (`VNDetectFaceLandmarksRequest`).
  - **Critical Difference**: `Vision` properties like `leftEye` refer to the **Person's Left Eye**. In a frontal photo, the Person's Left Eye is on the **Right** side of the image (Viewer's Right).

### 2. Alignment (Umeyama Helper)
- The `UmeyamaHelper` class expects 5 landmarks in a specific order corresponding to the reference points (112x112 standard):
  1.  **Viewer's Left Eye** (Low X)
  2.  **Viewer's Right Eye** (High X)
  3.  Nose
  4.  Viewer's Left Mouth Corner
  5.  Viewer's Right Mouth Corner

- **The Fix**: In `FaceLandmarkExtractor.swift`, we explicitly **swap** the Vision eye points before passing them to alignment:
  ```swift
  // Vision (Person's Right) -> Viewer's Left (Point 1)
  // Vision (Person's Left)  -> Viewer's Right (Point 2)
  let localPoints = [rEyeLocal, lEyeLocal, noseLocal, mouthLeftLocal, mouthRightLocal]
  ```

### 3. Normalization (Preprocessing)
- **Python**: `(x - 127.5) / 128.0` resulting in `[-1.0, 1.0]` range.
- **Swift**: The CoreML model input is a `CVPixelBuffer`.
  - **Important**: The CoreML model (`facenet_resnet.mlmodel`) **MUST** be exported with the normalization parameters baked in (scale=`1/128`, bias=`-127.5`). The Swift code passes raw BGRA pixel buffers; it does *not* manually normalize the pixel values in code. If the model expects `[0, 1]` or `[0, 255]`, recognition accuracy will fail.

### 4. Matching Logic & Thresholds
- **Distance Metric**: Cosine Similarity.
  - **Math Verification**: Verified in `MathUtilities.swift`. The implementation correctly uses `dotProduct / (magA * magB)`. It does NOT use `(cos + 1) / 2`.
  - **L2 Normalization**: The embeddings ARE L2-normalized in Swift before comparison (`MathUtilities.normalize`).
  - **Why High Scores?**: If input pixels are not normalized (0-255 range), the model outputs non-zero-centered embeddings. Even after L2 normalization, these vectors cluster in a small cone (positive quadrant), leading to artificially high cosine similarity (e.g., 0.6 instead of 0.0) for random faces. **The CoreML model fix is required.**

- **Threshold**: Defaults to `0.74` (configurable via `FaceRecognitionConfig`).
- **Logic**:
  1. Extract embedding (512-d).
  2. Compare against all gallery vectors using Cosine Similarity.
  3. Return the match with the highest score if it exceeds the threshold.

## Troubleshooting
- **Low Accuracy / False Positives**:
  - **Swapped Eyes**: Checked `FaceLandmarkExtractor`. Vision returns `leftEye` (Person's Left), but alignment expected Viewer's Left. **Fixed in `FaceLandmarkExtractor.swift`.**
  - **Input Normalization**: Checked CoreML input. If model expects -1..1 but gets 0..255, accuracy degrades and random scores increase. **Use `convert_torch_to_coreml.py`.**
- **"Unknown" Results**:
  - Check checks permissions and thresholds in `FaceNetFaceRecognitionEngine`.
