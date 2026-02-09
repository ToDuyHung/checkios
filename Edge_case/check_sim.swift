
import Foundation
import Vision
import CoreImage
import CoreML
import Accelerate

// MARK: - Helper Classes

/// A helper class to align faces using the Umeyama algorithm (Least Squares Similarity Transformation).
class UmeyamaHelper {
    private static let referenceLandmarks112: [CGPoint] = [
        CGPoint(x: 38.2946, y: 51.6963), // Viewer's Left Eye
        CGPoint(x: 73.5318, y: 51.5014), // Viewer's Right Eye
        CGPoint(x: 56.0252, y: 71.7366), // Nose
        CGPoint(x: 41.5493, y: 92.3655), // Viewer's Left Mouth
        CGPoint(x: 70.7299, y: 92.2041)  // Viewer's Right Mouth
    ]
    
    static func align(image: CIImage, landmarks: [CGPoint], outputSize: CGSize = CGSize(width: 160, height: 160)) -> CIImage? {
        guard landmarks.count == 5 else { return nil }
        
        let scaleX = outputSize.width / 112.0
        let scaleY = outputSize.height / 112.0
        
        let destPoints: [CGPoint] = referenceLandmarks112.map { point in
            let scaledX = point.x * scaleX
            let scaledY = point.y * scaleY
            // CoreImage origin is Bottom-Left, so flip Y
            return CGPoint(x: scaledX, y: outputSize.height - scaledY)
        }
        
        // Estimate Affine Transform
        guard let affineTransform = estimateAffine2D(from: landmarks, to: destPoints) else { return nil }
        
        // Apply Transform
        let alignedImage = image.transformed(by: affineTransform)
        
        // Crop
        let cropRect = CGRect(origin: .zero, size: outputSize)
        let croppedImage = alignedImage.cropped(to: cropRect)
        
        // Reset extent
        return croppedImage.transformed(by: CGAffineTransform(translationX: -croppedImage.extent.origin.x, y: -croppedImage.extent.origin.y))
    }
    
    private static func estimateAffine2D(from src: [CGPoint], to dst: [CGPoint]) -> CGAffineTransform? {
        // Simplified affine estimation (Translation + Scale + Rotation)
        // Since we don't have LAPACK/BLAS easy access without Accelerate complexities, 
        // We will assume standard Umeyama implementation behavior.
        // Or finding a simpler Least Squares fit.
        
        // NOTE: For this script to be standalone and accurate, implementing full SVD is heavy.
        // However, we can use a simplified similarity transform estimator if SVD is too much code.
        // BUT, to match Python exactly, we should use the same math.
        // Attempting to calculate means and covariance.
        
        let srcMean = mean(points: src)
        let dstMean = mean(points: dst)
        
        let num = Double(src.count)
        var srcDemean = src.map { CGPoint(x: $0.x - srcMean.x, y: $0.y - srcMean.y) }
        var dstDemean = dst.map { CGPoint(x: $0.x - dstMean.x, y: $0.y - dstMean.y) }
        
        // Covariance Matrix elements
        var s1: Double = 0, s2: Double = 0, s3: Double = 0, s4: Double = 0
        for i in 0..<src.count {
            s1 += Double(srcDemean[i].x * dstDemean[i].x + srcDemean[i].y * dstDemean[i].y)
            s2 += Double(srcDemean[i].x * dstDemean[i].y - srcDemean[i].y * dstDemean[i].x)
            s3 += Double(srcDemean[i].x * srcDemean[i].x + srcDemean[i].y * srcDemean[i].y)
            // s4 is for general affine, but here we constrain to Similarity
        }
        
        // Scale * cos(theta) = s1 / s3
        // Scale * sin(theta) = s2 / s3
        // If we assume just rotation/scale/translation
        
        // Let A = scale * cos(theta), B = scale * sin(theta)
        // A = s1 / s3
        // B = s2 / s3
        // However, standard Umeyama uses SVD to handle reflections and degenerate cases.
        // For face alignment, reflection is rare unless annotation is wrong.
        
        let A = s1 / s3
        let B = s2 / s3
        
        let scale = sqrt(A*A + B*B)
        // Rotation matrix elements derived from A and B
        
        let tx = Double(dstMean.x) - A * Double(srcMean.x) + B * Double(srcMean.y)
        let ty = Double(dstMean.y) - B * Double(srcMean.x) - A * Double(srcMean.y)
        
        return CGAffineTransform(a: CGFloat(A), b: CGFloat(-B), c: CGFloat(B), d: CGFloat(A), tx: CGFloat(tx), ty: CGFloat(ty))
    }
    
    private static func mean(points: [CGPoint]) -> CGPoint {
        let count = CGFloat(points.count)
        let sum = points.reduce(CGPoint.zero) { CGPoint(x: $0.x + $1.x, y: $0.y + $1.y) }
        return CGPoint(x: sum.x / count, y: sum.y / count)
    }
}

class FaceLandmarkExtractor {
    static func extractFivePoints(from observation: VNFaceObservation, imageSize: CGSize) -> [CGPoint]? {
        guard let landmarks = observation.landmarks,
              let leftEye = landmarks.leftEye,
              let rightEye = landmarks.rightEye,
              let nose = landmarks.nose,
              let lips = landmarks.outerLips else { return nil }
        
        func centroid(_ region: VNFaceLandmarkRegion2D) -> CGPoint {
            let pts = region.normalizedPoints
            let sum = pts.reduce(CGPoint.zero) { CGPoint(x: $0.x + $1.x, y: $0.y + $1.y) }
            return CGPoint(x: sum.x / CGFloat(pts.count), y: sum.y / CGFloat(pts.count))
        }
        
        // Vision: LeftEye is Person's Left (Viewer's Right)
        let lEyeLocal = centroid(leftEye) // Viewer's Right
        let rEyeLocal = centroid(rightEye) // Viewer's Left
        let noseLocal = centroid(nose)
        
        let mouthLeftLocal = lips.normalizedPoints.min(by: { $0.x < $1.x }) ?? .zero
        let mouthRightLocal = lips.normalizedPoints.max(by: { $0.x < $1.x }) ?? .zero
        
        // Umeyama Expects: [Viewer's Left Eye, Viewer's Right Eye, Nose, Viewer's Left Mouth, Viewer's Right Mouth]
        // So we swap eyes: [rEyeLocal, lEyeLocal, ...]
        let points = [rEyeLocal, lEyeLocal, noseLocal, mouthLeftLocal, mouthRightLocal]
        
        // Vision bbox is normalized relative to image. (0,0) is bottom-left?
        // VNFaceObservation bbox origin IS Bottom-Left in normalized coordinates (0..1).
        
        let bbox = observation.boundingBox
        return points.map { p in
            // p is relative to bbox
            let x = bbox.origin.x + p.x * bbox.size.width
            let y = bbox.origin.y + p.y * bbox.size.height
            return CGPoint(x: x * imageSize.width, y: y * imageSize.height)
        }
    }
    
    static func extractLandmarks(from image: CIImage) -> [CGPoint]? {
        let handler = VNImageRequestHandler(ciImage: image, options: [:])
        let req = VNDetectFaceLandmarksRequest()
        try? handler.perform([req])
        guard let res = req.results?.first else { return nil }
        return extractFivePoints(from: res, imageSize: image.extent.size)
    }
}

// MARK: - Main Logic

func compileModel(at url: URL) -> URL? {
    let compileProc = Process()
    compileProc.executableURL = URL(fileURLWithPath: "/usr/bin/xcrun")
    
    let tempDir = FileManager.default.temporaryDirectory
    compileProc.arguments = ["coremlcompiler", "compile", url.path, tempDir.path]
    
    try? compileProc.run()
    compileProc.waitUntilExit()
    
    if compileProc.terminationStatus == 0 {
        let compiledName = url.deletingPathExtension().appendingPathExtension("mlmodelc").lastPathComponent
        return tempDir.appendingPathComponent(compiledName)
    }
    return nil
}

func l2_normalize(_ vector: [Float]) -> [Float] {
    let sumSq = vector.reduce(0) { $0 + $1 * $1 }
    let norm = sqrt(sumSq)
    if norm < 1e-6 { return vector }
    return vector.map { $0 / norm }
}

func cosine_sim(_ a: [Float], _ b: [Float]) -> Float {
    let dot = zip(a, b).map(*).reduce(0, +)
    // Assuming inputs already normalized
    return dot
}

struct FaceRecord {
    let id: String
    let embedding: [Float]
}

func main() {
    let fileManager = FileManager.default
    let currentDir = URL(fileURLWithPath: fileManager.currentDirectoryPath)
    
    // Paths
    let edgeCaseDir = currentDir.appendingPathComponent("Edge_case")
    let validDir = edgeCaseDir.appendingPathComponent("valid")
    let galleryDir = edgeCaseDir.appendingPathComponent("gallery")
    let modelPath = currentDir.appendingPathComponent("onnx_models/FaceNet.mlpackage")
    
    print("Checking paths...")
    print("Valid Dir: \(validDir.path)")
    print("Gallery Dir: \(galleryDir.path)")
    print("Model Path: \(modelPath.path)")
    
    guard fileManager.fileExists(atPath: modelPath.path) else {
        print("Error: FaceNet.mlpackage not found at \(modelPath.path)")
        exit(1)
    }
    
    print("Compiling CoreML Model...")
    guard let compiledUrl = compileModel(at: modelPath) else {
        print("Error: Failed to compile model.")
        exit(1)
    }
    
    print("Loading CoreML Model...")
    guard let model = try? MLModel(contentsOf: compiledUrl) else {
        print("Error: Failed to load compiled model.")
        exit(1)
    }
    
    // Function to process image
    func getEmbedding(path: URL) -> [Float]? {
        guard let image = CIImage(contentsOf: path) else { return nil }
        
        // Detect & Align
        guard let landmarks = FaceLandmarkExtractor.extractLandmarks(from: image),
              let aligned = UmeyamaHelper.align(image: image, landmarks: landmarks) else {
            return nil
        }
        
        // Create Pixel Buffer
        let size = CGSize(width: 160, height: 160)
        var pixelBuffer: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, 160, 160, kCVPixelFormatType_32BGRA, 
                            [kCVPixelBufferCGImageCompatibilityKey: true, kCVPixelBufferCGBitmapContextCompatibilityKey: true] as CFDictionary, 
                            &pixelBuffer)
        
        guard let buffer = pixelBuffer else { return nil }
        let context = CIContext()
        context.render(aligned, to: buffer) // CIImage handles scaling if implicit? No, align output is 160x160.
        
        // Run Model
        // Identify input name: usually "input" or "image"
        let inputName = model.modelDescription.inputDescriptionsByName.keys.first ?? "input"
        
        guard let input = try? MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(pixelBuffer: buffer)]),
              let output = try? model.prediction(from: input) else {
            return nil
        }
        
        // Get Output
        // Identify output name
        let outputName = model.modelDescription.outputDescriptionsByName.keys.first ?? "embedding"
        guard let multiArray = output.featureValue(for: outputName)?.multiArrayValue else { return nil }
        
        var embedding = [Float](repeating: 0, count: 512)
        for i in 0..<512 {
            embedding[i] = multiArray[i].floatValue
        }
        
        return l2_normalize(embedding)
    }
    
    // Load Gallery
    print("Loading Gallery...")
    var gallery: [FaceRecord] = []
    
    if let galleryFiles = try? fileManager.contentsOfDirectory(at: galleryDir, includingPropertiesForKeys: nil) {
        for file in galleryFiles {
            if ["jpg", "png", "jpeg"].contains(file.pathExtension.lowercased()) {
                if let emb = getEmbedding(path: file) {
                    let id = file.deletingPathExtension().lastPathComponent
                    gallery.append(FaceRecord(id: id, embedding: emb))
                    print("Loaded gallery id: \(id)")
                }
            }
        }
    }
    
    print("Loaded \(gallery.count) gallery faces.")
    
    // Process Valid
    print("\nProcessing Valid Images...")
    if let validFiles = try? fileManager.subpathsOfDirectory(atPath: validDir.path) {
        for file in validFiles {
             let fullPath = validDir.appendingPathComponent(file)
             if ["jpg", "png", "jpeg"].contains(fullPath.pathExtension.lowercased()) {
                 print("Processing: \(file)")
                 if let emb = getEmbedding(path: fullPath) {
                     var maxSim: Float = -1.0
                     var bestID = "Unknown"
                     
                     for rec in gallery {
                         let sim = cosine_sim(emb, rec.embedding)
                         if sim > maxSim {
                             maxSim = sim
                             bestID = rec.id
                         }
                     }
                     print("  Max Similarity: \(String(format: "%.4f", maxSim)) (Matched with \(bestID))")
                 } else {
                     print("  No face detected or embedding failed.")
                 }
             }
        }
    }
}

// Run
if #available(macOS 12.0, *) {
    main()
} else {
    print("Requires macOS 12.0+")
}
