//
//  FaceLandmarkExtractor.swift
//  VCCSmartOffice
//
//  Created by Doan Thi Minh Hoa on 9/2/26.
//

import Vision
import CoreImage
import UIKit

/// A helper class to extract facial landmarks from a VNFaceObservation.
/// It extracts the 5 key points required for face alignment:
/// Left Eye, Right Eye, Nose, Left Mouth Corner, Right Mouth Corner.
public class FaceLandmarkExtractor {
    
    /// Extracts 5 key landmarks from a face observation, normalized to the image size.
    ///
    /// - Parameters:
    ///   - observation: The `VNFaceObservation` containing landmarks.
    ///   - imageSize: The size of the image the observation belongs to.
    /// - Returns: An array of 5 `CGPoint` in the order: [Left Eye, Right Eye, Nose, Left Mouth, Right Mouth],
    ///            scaled to the `imageSize`. Returns `nil` if any required landmark is missing.
    public static func extractFivePoints(from observation: VNFaceObservation, imageSize: CGSize) -> [CGPoint]? {
        guard let landmarks = observation.landmarks else { return nil }
        
        let bbox = observation.boundingBox
        
        // Ensure all required landmarks are present
        guard let leftEyePoints = landmarks.leftEye?.normalizedPoints, !leftEyePoints.isEmpty,
              let rightEyePoints = landmarks.rightEye?.normalizedPoints, !rightEyePoints.isEmpty,
              let nosePoints = landmarks.nose?.normalizedPoints, !nosePoints.isEmpty,
              let outerLips = landmarks.outerLips?.normalizedPoints, !outerLips.isEmpty else {
            return nil
        }
        
        // Helper to compute centroid of points
        func centroid(_ points: [CGPoint]) -> CGPoint {
            let sum = points.reduce(CGPoint.zero) { CGPoint(x: $0.x + $1.x, y: $0.y + $1.y) }
            return CGPoint(x: sum.x / CGFloat(points.count), y: sum.y / CGFloat(points.count))
        }
        
        // Compute centroids for eyes and nose
        let lEyeLocal = centroid(leftEyePoints)
        let rEyeLocal = centroid(rightEyePoints)
        let noseLocal = centroid(nosePoints)
        
        // Compute mouth corners from outer lips
        // Left mouth corner has the smallest x, Right mouth corner has the largest x
        // Note: In normalized coordinates (0,0 is bottom-left for Vision?), check coordinate system.
        // Vision Normalized: (0,0) is Bottom-Left.
        // However, standard "Left" means the person's left (viewer's right).
        // But here we likely mean "Left side of image" vs "Right side of image".
        // Let's stick to existing logic from FaceNetFaceRecognitionEngine:
        // let mouthLeftLocal = outerLips.min(by: { $0.x < $1.x })
        // This gives the point with smallest X (Viewer's Left).
        
        let mouthLeftLocal = outerLips.min(by: { $0.x < $1.x }) ?? CGPoint.zero
        let mouthRightLocal = outerLips.max(by: { $0.x < $1.x }) ?? CGPoint.zero
        
        // Order: Left Eye, Right Eye, Nose, Left Mouth, Right Mouth
        // Note: Check if "Left Eye" means the eye on the left of the image (Person's Right Eye)?
        // In Vision, `leftEye` is the person's left eye (viewer's right).
        // `FaceNetFaceRecognitionEngine` likely used Vision's `leftEye` directly.
        // Let's verify existing implementation logic.
        
        // In previously viewed code:
        // let lEyeLocal = centroid(leftEyePoints) -> from landmarks.leftEye
        // landmarks.leftEye is Person's Left Eye (Viewer's Right).
        
        // However, standard implementation usually expects points in order of X-coordinate or specific anatomical points.
        // If Model expects 5 points, usually it's [Left Eye, Right Eye, ...].
        // "Left Eye" usually implies the eye on the left side of the image (Person's Right Eye).
        // But `VNFaceLandmarks2D.leftEye` property documentation says "The region containing the left eye." (Person's Left).
        
        // Let's check `FaceNetFaceRecognitionEngine` original implementation again.
        // It extracted `leftEyePoints` from `landmarks.leftEye`.
        // Then returned `[lEyeLocal, rEyeLocal, noseLocal, mouthLeftLocal, mouthRightLocal]`.
        // So I will preserve this order.
        
        // Fix: Use Viewer's Perspective for Umeyama Alignment
        // Vision "leftEye" is Person's Left (Viewer's Right).
        // Vision "rightEye" is Person's Right (Viewer's Left).
        // Umeyama Reference expects: [Viewer's Left, Viewer's Right, ...]
        let localPoints = [rEyeLocal, lEyeLocal, noseLocal, mouthLeftLocal, mouthRightLocal]
        
        return localPoints.map { p in
            // Convert normalized point (within bounding box) to image coordinates
            // Vision bbox origin is Bottom-Left.
            // p is normalized within bbox (0..1).
            // imageSize is pixels.
            
            let xNorm = bbox.origin.x + p.x * bbox.size.width
            let yNorm = bbox.origin.y + p.y * bbox.size.height
            
            return CGPoint(x: xNorm * imageSize.width, y: yNorm * imageSize.height)
        }
    }
    
    /// Extracts 5 key landmarks from a `CIImage`.
    /// Performs face detection and selects the largest face if multiple are found.
    ///
    /// - Parameter image: The `CIImage` to process.
    /// - Returns: An array of 5 `CGPoint` landmarks, or `nil` if detection fails.
    public static func extractLandmarks(from image: CIImage) -> [CGPoint]? {
        let handler = VNImageRequestHandler(ciImage: image, options: [:])
        let faceRequest = VNDetectFaceLandmarksRequest()
        
        do {
            try handler.perform([faceRequest])
            
            guard var faceObservation = faceRequest.results?.first else {
                return nil
            }
            
            if let results = faceRequest.results, results.count > 1 {
                if let largest = results.max(by: { $0.boundingBox.width * $0.boundingBox.height < $1.boundingBox.width * $1.boundingBox.height }) {
                    faceObservation = largest
                }
            }
            
            return extractFivePoints(from: faceObservation, imageSize: image.extent.size)
        } catch {
            print("FaceLandmarkExtractor: Error detecting faces - \(error)")
            return nil
        }
    }
}
