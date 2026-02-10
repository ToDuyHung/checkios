import CoreImage
import Accelerate
import Foundation

/// A helper class to align faces using the Umeyama algorithm (Least Squares Similarity Transformation).
/// Ported from `onnx_baseline.py`.
///
/// This class handles the coordinate system differences between standard computer vision libraries (Top-Left origin)
/// and Core Image (Bottom-Left origin).
public class UmeyamaHelper {
    
    // MARK: - Constants
    
    private static let minSingularValue = 1e-12
    private static let varianceEpsilon = 1e-12
    
    /// Standard 5 facial landmarks for a 112x112 image in Top-Left coordinates.
    /// Order: Left Eye, Right Eye, Nose, Left Mouth Corner, Right Mouth Corner.
    private static let referenceLandmarks112: [CGPoint] = [
        CGPoint(x: 38.2946, y: 51.6963),
        CGPoint(x: 73.5318, y: 51.5014),
        CGPoint(x: 56.0252, y: 71.7366),
        CGPoint(x: 41.5493, y: 92.3655),
        CGPoint(x: 70.7299, y: 92.2041)
    ]
    
    // MARK: - Public API
    
    /// Aligns a face in a CIImage based on the provided 5 landmarks.
    ///
    /// - Parameters:
    ///   - image: The source `CIImage` to align.
    ///   - landmarks: An array of 5 `CGPoint`s representing the detected facial landmarks.
    ///                These must be in the same coordinate space as the `image` (Bottom-Left origin for CIImage).
    ///                Order must be: Left Eye, Right Eye, Nose, Left Mouth, Right Mouth.
    ///   - outputSize: The desired size of the output aligned image. Default is 160x160.
    /// - Returns: A new `CIImage` aligned to the standard face position, or `nil` if alignment fails.
    public static func align(image: CIImage, landmarks: [CGPoint], outputSize: CGSize = CGSize(width: 160, height: 160)) -> CIImage? {
        guard landmarks.count == 5 else {
            print("UmeyamaHelper: Expected 5 landmarks, got \(landmarks.count).")
            return nil
        }
        
        guard landmarks.allSatisfy({ $0.x.isFinite && $0.y.isFinite }) else {
            print("UmeyamaHelper: Invalid landmark coordinates (NaN or Inf).")
            return nil
        }
        
        // 1. Prepare Reference Landmarks scaled to output size
        let scaleX = outputSize.width / 112.0
        let scaleY = outputSize.height / 112.0
        
        let destPoints: [CGPoint] = referenceLandmarks112.map { point in
            let scaledX = point.x * scaleX
            let scaledY = point.y * scaleY
            return CGPoint(x: scaledX, y: outputSize.height - scaledY)
        }
        
        // 2. Compute the Similarity Transform Matrix
        guard let affineTransform = estimateAffine2D(from: landmarks, to: destPoints) else {
            return nil
        }
        
        // 3. Apply the Transform
        let alignedImage = image.transformed(by: affineTransform)
        
        // 4. Crop to the desired output size
        let cropRect = CGRect(origin: .zero, size: outputSize)
        let croppedImage = alignedImage.cropped(to: cropRect)
        
        // Reset the extent to origin (0,0)
        let finalImage = croppedImage.transformed(by: CGAffineTransform(translationX: -croppedImage.extent.origin.x, y: -croppedImage.extent.origin.y))
        
        return finalImage
    }
    
    // MARK: - Private Helpers (Umeyama Implementation)
    
    /// Computes the similarity transform (Rotation, Scale, Translation) from `src` points to `dst` points.
    /// M minimizes the sum of squared errors: sum || (s * R * src_i + t) - dst_i ||^2
    private static func estimateAffine2D(from src: [CGPoint], to dst: [CGPoint]) -> CGAffineTransform? {
        let num = src.count
        guard num == dst.count, num > 0 else { return nil }
        
        // 1. Compute Centroids
        let srcMean = mean(points: src)
        let dstMean = mean(points: dst)
        
        // 2. Center the points (Demean) - Pre-allocate for performance
        var srcDemean = [SIMD2<Double>]()
        var dstDemean = [SIMD2<Double>]()
        srcDemean.reserveCapacity(num)
        dstDemean.reserveCapacity(num)
        
        for i in 0..<num {
            srcDemean.append(SIMD2<Double>(Double(src[i].x - srcMean.x), Double(src[i].y - srcMean.y)))
            dstDemean.append(SIMD2<Double>(Double(dst[i].x - dstMean.x), Double(dst[i].y - dstMean.y)))
        }
        
        // 3. Compute Covariance Matrix A = (dst_demean^T * src_demean) / num
        var A = matrix2x2_zeros()
        for i in 0..<num {
            A.0 += dstDemean[i].x * srcDemean[i].x
            A.1 += dstDemean[i].x * srcDemean[i].y
            A.2 += dstDemean[i].y * srcDemean[i].x
            A.3 += dstDemean[i].y * srcDemean[i].y
        }
        A.0 /= Double(num)
        A.1 /= Double(num)
        A.2 /= Double(num)
        A.3 /= Double(num)
        
        // 4. SVD of A
        var aFlat = [A.0, A.2, A.1, A.3]
        var m: Int32 = 2
        var n: Int32 = 2
        var lda: Int32 = 2
        var s = [Double](repeating: 0, count: 2)
        var u = [Double](repeating: 0, count: 4) // 2x2
        var ldu: Int32 = 2
        var vt = [Double](repeating: 0, count: 4) // 2x2
        var ldvt: Int32 = 2
        
        // Query optimal work array size
        var lwork: Int32 = -1
        var work = [Double](repeating: 0, count: 1)
        var info: Int32 = 0
        var jobu: Int8 = 65 // 'A'
        var jobvt: Int8 = 65 // 'A'
        
        dgesvd_(&jobu, &jobvt, &m, &n, &aFlat, &lda, &s, &u, &ldu, &vt, &ldvt, &work, &lwork, &info)
        lwork = Int32(work[0])
        work = [Double](repeating: 0, count: Int(lwork))
        
        // Perform actual SVD
        dgesvd_(&jobu, &jobvt, &m, &n, &aFlat, &lda, &s, &u, &ldu, &vt, &ldvt, &work, &lwork, &info)
        
        guard info == 0 else {
            print("UmeyamaHelper: SVD failed with info \(info)")
            return nil
        }
        
        // Check for singular matrix
        guard s[0] > minSingularValue else {
            print("UmeyamaHelper: Singular matrix detected")
            return nil
        }
        
        // 5. Compute Rotation R = U * Vt
        func get(_ arr: [Double], _ r: Int, _ c: Int) -> Double { return arr[c * 2 + r] }
        
        var R = matrix2x2_zeros()
        R.0 = get(u, 0, 0) * get(vt, 0, 0) + get(u, 0, 1) * get(vt, 1, 0) // R00
        R.1 = get(u, 0, 0) * get(vt, 0, 1) + get(u, 0, 1) * get(vt, 1, 1) // R01
        R.2 = get(u, 1, 0) * get(vt, 0, 0) + get(u, 1, 1) * get(vt, 1, 0) // R10
        R.3 = get(u, 1, 0) * get(vt, 0, 1) + get(u, 1, 1) * get(vt, 1, 1) // R11
        
        // 6. Check Determinant and adjust if reflection
        let detR = R.0 * R.3 - R.1 * R.2
        if detR < 0 {
            // Flip last column of U
            u[2] *= -1
            u[3] *= -1
            
            // Recompute R
            R.0 = get(u, 0, 0) * get(vt, 0, 0) + get(u, 0, 1) * get(vt, 1, 0)
            R.1 = get(u, 0, 0) * get(vt, 0, 1) + get(u, 0, 1) * get(vt, 1, 1)
            R.2 = get(u, 1, 0) * get(vt, 0, 0) + get(u, 1, 1) * get(vt, 1, 0)
            R.3 = get(u, 1, 0) * get(vt, 0, 1) + get(u, 1, 1) * get(vt, 1, 1)
        }
        
        // 7. Scale
        var varSrc: Double = 0
        for i in 0..<num {
            varSrc += srcDemean[i].x * srcDemean[i].x + srcDemean[i].y * srcDemean[i].y
        }
        varSrc /= Double(num)
        
        let scale = (s[0] + s[1]) / (varSrc + varianceEpsilon)
        
        // 8. Translation
        let rSrcMeanX = R.0 * Double(srcMean.x) + R.1 * Double(srcMean.y)
        let rSrcMeanY = R.2 * Double(srcMean.x) + R.3 * Double(srcMean.y)
        
        let tX = Double(dstMean.x) - scale * rSrcMeanX
        let tY = Double(dstMean.y) - scale * rSrcMeanY
        
        // 9. Construct Affine Transform
        return CGAffineTransform(
            a: CGFloat(scale * R.0),
            b: CGFloat(scale * R.2),
            c: CGFloat(scale * R.1),
            d: CGFloat(scale * R.3),
            tx: CGFloat(tX),
            ty: CGFloat(tY)
        )
    }
    
    // MARK: - Utilities
    
    private typealias Matrix2x2 = (Double, Double, Double, Double)
    
    private static func matrix2x2_zeros() -> Matrix2x2 {
        return (0, 0, 0, 0)
    }
    
    private static func mean(points: [CGPoint]) -> CGPoint {
        var x: CGFloat = 0
        var y: CGFloat = 0
        for p in points {
            x += p.x
            y += p.y
        }
        let count = CGFloat(points.count)
        return CGPoint(x: x / count, y: y / count)
    }
}
