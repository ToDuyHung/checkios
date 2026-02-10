
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
        let srcMean = mean(points: src)
        let dstMean = mean(points: dst)
        
        let num = Double(src.count)
        var srcDemean = src.map { CGPoint(x: $0.x - srcMean.x, y: $0.y - srcMean.y) }
        var dstDemean = dst.map { CGPoint(x: $0.x - dstMean.x, y: $0.y - dstMean.y) }
        
        var s1: Double = 0, s2: Double = 0, s3: Double = 0
        for i in 0..<src.count {
            s1 += Double(srcDemean[i].x * dstDemean[i].x + srcDemean[i].y * dstDemean[i].y)
            s2 += Double(srcDemean[i].x * dstDemean[i].y - srcDemean[i].y * dstDemean[i].x)
            s3 += Double(srcDemean[i].x * srcDemean[i].x + srcDemean[i].y * srcDemean[i].y)
        }
        
        let A = s1 / s3
        let B = s2 / s3
        
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
        
        let lEyeLocal = centroid(leftEye)
        let rEyeLocal = centroid(rightEye)
        let noseLocal = centroid(nose)
        
        let mouthLeftLocal = lips.normalizedPoints.min(by: { $0.x < $1.x }) ?? .zero
        let mouthRightLocal = lips.normalizedPoints.max(by: { $0.x < $1.x }) ?? .zero
        
        // Umeyama Expects: [Viewer's Left Eye, Viewer's Right Eye, Nose, Viewer's Left Mouth, Viewer's Right Mouth]
        // Vision LeftEye = Person's Left = Viewer's Right.
        // So we swap: [rightEye, leftEye...]
        let points = [rEyeLocal, lEyeLocal, noseLocal, mouthLeftLocal, mouthRightLocal]
        
        let bbox = observation.boundingBox
        return points.map { p in
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

func main() {
    // Arguments: [script, image_path]
    let args = CommandLine.arguments
    guard args.count > 1 else {
        print("Usage: swift embed.swift <image_path>")
        exit(1)
    }
    
    let imagePath = URL(fileURLWithPath: args[1])
    let fileManager = FileManager.default
    let currentDir = URL(fileURLWithPath: fileManager.currentDirectoryPath)
    let modelPath = currentDir.appendingPathComponent("onnx_models/FaceNet.mlpackage")
    
    guard fileManager.fileExists(atPath: imagePath.path) else {
        print("Error: Image not found at \(imagePath.path)")
        exit(1)
    }
    
    guard fileManager.fileExists(atPath: modelPath.path) else {
        print("Error: Model not found at \(modelPath.path)")
        exit(1)
    }
    
    // Compile Model
    guard let compiledUrl = compileModel(at: modelPath) else {
        print("Error: Failed to compile model")
        exit(1)
    }
    
    // Load Model
    guard let model = try? MLModel(contentsOf: compiledUrl) else {
        print("Error: Failed to load model")
        exit(1)
    }
    
    // Process Image
    guard let image = CIImage(contentsOf: imagePath) else {
        print("Error: Failed to load CIImage")
        exit(1)
    }
    
    // Detect & Align
    guard let landmarks = FaceLandmarkExtractor.extractLandmarks(from: image),
          let aligned = UmeyamaHelper.align(image: image, landmarks: landmarks) else {
        print("Error: Face detection/alignment failed")
        exit(1)
    }
    
    // Create Pixel Buffer (160x160)
    var pixelBuffer: CVPixelBuffer?
    CVPixelBufferCreate(kCFAllocatorDefault, 160, 160, kCVPixelFormatType_32BGRA, 
                        [kCVPixelBufferCGImageCompatibilityKey: true, kCVPixelBufferCGBitmapContextCompatibilityKey: true] as CFDictionary, 
                        &pixelBuffer)
    
    guard let buffer = pixelBuffer else {
        print("Error: Failed to create PixelBuffer")
        exit(1)
    }
    
    // CRITICAL FIX: Disable Color Management
    // Use NSNull for working/output color space to treat image as RAW bytes.
    // This matches Python/OpenCV behavior which ignores ICC profiles/Gamma.
    let context = CIContext(options: [
        .workingColorSpace: NSNull(), 
        .outputColorSpace: NSNull()
    ])
    
    context.render(aligned, to: buffer)
    
    // DEBUG: Save aligned image for verification
    let debugPath = currentDir.appendingPathComponent("debug_embed_aligned.jpg")
    if let cgImage = context.createCGImage(aligned, from: aligned.extent) {
        let uiImage = UIImage(cgImage: cgImage)
        if let data = uiImage.jpegData(compressionQuality: 1.0) {
            try? data.write(to: debugPath)
            print("Saved debug image: \(debugPath.path)")
        }
    }
    
    // Run Model
    let inputName = model.modelDescription.inputDescriptionsByName.keys.first ?? "input"
    guard let input = try? MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(pixelBuffer: buffer)]),
          let output = try? model.prediction(from: input) else {
        print("Error: Model inference failed")
        exit(1)
    }
    
    // Output
    let outputName = model.modelDescription.outputDescriptionsByName.keys.first ?? "embedding"
    guard let multiArray = output.featureValue(for: outputName)?.multiArrayValue else {
        print("Error: Invalid output format")
        exit(1)
    }
    
    var embedding = [Float](repeating: 0, count: 512)
    for i in 0..<512 {
        embedding[i] = multiArray[i].floatValue
    }
    
    let normalized = l2_normalize(embedding)
    
    // Print Vector space-separated (for easy copy-paste/comparison)
    let vectorStr = normalized.map { String($0) }.joined(separator: " ")
    print(vectorStr)
}

if #available(macOS 12.0, *) {
    main()
} else {
    print("Requires macOS 12.0+")
}
