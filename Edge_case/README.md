# Edge Case Analysis: Swift vs Python CoreML Discrepancy

## Status
- **Model**: `FaceNet.mlpackage` (Converted from PyTorch with `scale=1/128`, `bias=-127.5`, `RGB` layout).
- **Python Inference**: Matches PyTorch exactly.
- **Swift Inference**: Differs from Python.

## Why the Discrepancy?
Since the model is identical and verified correct in Python, the issue lies in **Input Preprocessing**.
The definition of "Input" includes:
1.  **Face Alignment (Umeyama)**: Swift implementation vs Python implementation.
2.  **Image Resizing/Interpolation**: CoreImage vs OpenCV/PIL.
3.  **Pixel Format**: BGRA vs RGB.

### 1. Face Alignment Differences
Swift uses `UmeyamaHelper.swift` (a port), while Python uses `facenet-pytorch` or `opencv`.
Even small differences in landmark coordinates (Vision vs MTCNN) or the affine transform calculation will result in slightly different pixels in the 160x160 aligned image.
**This is the most likely cause.**
-   **Vision Landmarks**: Normalized 0..1, based on bounding box.
-   **MTCNN Landmarks**: Absolute pixel coordinates.
-   **Transform Solver**: Swift implementation might have minor precision differences.

### 3. Pixel Format & Color Space (CRITICAL)
-   **Python**: OpenCV loads images as Raw BGR/RGB values exactly as stored in the file. It does **not** apply Gamma correction or ICC profiles.
-   **Swift**: `CIContext` by default applies Color Management (sRGB Gamma, Device Profiles). This changes the pixel values (e.g., 100 becomes 105), changing the vector significantly.
-   **Fix**: Initialize `CIContext` with `NSNull` for color spaces to disable management:
    ```swift
    let context = CIContext(options: [
        .workingColorSpace: NSNull(), 
        .outputColorSpace: NSNull()
    ])
    ```
    This forces `render` to copy raw bits, matching Python.

## Solution Strategy
To make Swift output match Python, we must align the *Preprocessing Pipeline*, not just the model.
1.  **Verify Aligned Images**: Save the 160x160 aligned image from Swift (`debug_image.jpg`) and compare it with the aligned image from Python.
    -   If they look different (shifted, rotated, zoomed), **fix `UmeyamaHelper.swift`**.
2.  **Verify Color Space**: Ensure Swift is creating `kCVPixelFormatType_32BGRA` (standard for iOS/macOS) which CoreML handles, OR manually create RGB data.
3.  **Accept Tolerance**: It is *normal* for different platforms (Vision/CI vs OpenCV) to have slight numerical differences. If the similarity between "Swift Vector" and "Python Vector" is > 0.90, it is usually acceptable.
    -   **Current Status**: User reported 1.0 (likely bug) or significant difference.
    -   **Goal**: If vector sim is > 0.95, it is fine. If < 0.9, we need to fix alignment.
