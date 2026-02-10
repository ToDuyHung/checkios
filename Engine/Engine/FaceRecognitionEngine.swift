//
//  FaceRecognitionEngine.swift
//  VCCSmartOffice
//
//  Created By Leo on 1/12/25.
//

import Foundation
import CoreImage

// MARK: - FaceMatchResult

struct FaceMatchResult {
    let matchedFace: FaceData
    let capturedImageData: Data?
    let confidence: Float
}

// MARK: - FaceQualityResult

struct FaceQualityResult {
    let quality: Float
    let hasFace: Bool
    var faceImage: CIImage? = nil
    
    // Detailed Metrics (Optional)
    var faceSizeRatio: Float? = nil
    var brightness: Float? = nil
    var blurScore: Float? = nil
    var yaw: Float? = nil
    var roll: Float? = nil
    var pitch: Float? = nil
}

// MARK: - FaceRecognitionEngine Protocol

protocol FaceRecognitionEngine {
    // MARK: - Registration

    func registerFaceWithSamples(_ samples: [[Float]], id: FaceIDType, userName: String, userEmail: String?) throws -> FaceIDType
    
    func extractEmbedding(from image: CIImage) async throws -> [Float]
    
    func extractEmbeddingAndImage(from image: CIImage) async throws -> (embedding: [Float], alignedImage: CIImage)

    // MARK: - Verification

    func verifyFace(from image: CIImage) async throws -> FaceMatchResult

    // MARK: - Management

    func getAllRegisteredFaces() -> [FaceData]

    func deleteFace(id: FaceIDType) throws

    func deleteAllFaces()
}

