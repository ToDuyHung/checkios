//
//  FaceNetFaceRecognitionEngine.swift
//  VCCSmartOffice
//
//  Created By Leo on 8/12/25.
//

import Foundation
import Vision
import CoreImage
import CoreML
import Accelerate

// MARK: - PyTorch Face Recognition Engine

final class FaceNetFaceRecognitionEngine: FaceRecognitionEngine {
    
    // MARK: Properties
    
    private let mlModel: MLModel
    private let minimumQuality: Float
    private let embeddingSize: Int
    private let modelInputSize: CGSize
    
    private let repository: FaceIDRepository
    
    // MARK: Initialization
    
    init(
        mlModel: MLModel,
        embeddingSize: Int = FaceRecognitionConfig.Model.embeddingSize,
        modelInputSize: CGSize = FaceRecognitionConfig.Model.inputSize,
        minimumQuality: Float = FaceRecognitionConfig.Quality.faceQualityThreshold,
        repository: FaceIDRepository = FaceIDRepository()
    ) throws {
        self.mlModel = mlModel
        self.embeddingSize = embeddingSize
        self.modelInputSize = modelInputSize
        self.minimumQuality = minimumQuality
        self.repository = repository
    }
    
    // MARK: Multi-Sample Registration

    func registerFaceWithSamples(_ samples: [[Float]], id: FaceIDType, userName: String, userEmail: String?) throws -> FaceIDType {
        for sample in samples {
            let record = FaceVectorRecord(
                id: UUID(),
                userID: id,
                userName: userName,
                userEmail: userEmail,
                vector: sample,
                createdAt: Date(),
            )
            repository.saveRecord(record)
        }
        
        return id
    }

    // MARK: Face Verification

    func verifyFace(from image: CIImage) async throws -> FaceMatchResult {
        let allRecords = repository.getAllRecords()

        guard !allRecords.isEmpty else {
            throw FacialRecognitionError.noRegisteredFaces
        }

        let queryEmbedding = try await extractEmbedding(from: image)

        var bestMatch: (record: FaceVectorRecord, similarity: Float)?

        for record in allRecords {
            let similarity = calculateCosineSimilarity(queryEmbedding, record.vector)
            
            if let current = bestMatch {
                if similarity > current.similarity {
                    bestMatch = (record, similarity)
                }
            } else {
                bestMatch = (record, similarity)
            }
        }

        let currentThreshold = FaceRecognitionConfigService.shared.getSimilarityThreshold()
        
        guard let match = bestMatch, match.similarity >= currentThreshold else {
            throw FacialRecognitionError.faceMismatch(theBestSimiliarity: bestMatch?.similarity ?? 0.0)
        }

        // Convert CIImage to Data using ImageHelper
        let imageData = ImageHelper.ciImageToData(image)

        let faceData = FaceData(
            id: match.record.userID,
            userName: match.record.userName,
            userEmail: match.record.userEmail,
            faceDescriptor: match.record.vector,
            registrationDate: match.record.createdAt,
        )

        return FaceMatchResult(
            matchedFace: faceData,
            capturedImageData: imageData,
            confidence: match.similarity
        )
    }
    
    // MARK: Face Management

    func getAllRegisteredFaces() -> [FaceData] {
        let records = repository.getAllRecords()
        
        let grouped = Dictionary(grouping: records, by: { $0.userID })
        
        return grouped.map { (userID, userRecords) -> FaceData in
            let representative = userRecords.max(by: { $0.createdAt < $1.createdAt })!
            
            return FaceData(
                id: userID,
                userName: representative.userName,
                userEmail: representative.userEmail,
                faceDescriptor: representative.vector,
                registrationDate: representative.createdAt
            )
        }
    }

    func deleteFace(id: FaceIDType) throws {
        let records = repository.getAllRecords().filter { $0.userID == id }
        guard !records.isEmpty else {
             throw FacialRecognitionError.storageError
        }
        
        repository.deleteRecords(for: id)
    }

    func deleteAllFaces() {
        repository.deleteAll()
    }
    
    // MARK: Private - Embedding Extraction

    private func alignFace(for image: CIImage) throws -> CIImage {
        guard let landmarks = FaceLandmarkExtractor.extractLandmarks(from: image) else {
            print("⚠️ Failed to extract landmarks.")
            throw FacialRecognitionError.noFaceDetected
        }

        guard let alignedImage = UmeyamaHelper.align(image: image, landmarks: landmarks, outputSize: modelInputSize) else {
            print("⚠️ Umeyama alignment failed.")
            throw FacialRecognitionError.processingFailed
        }
        
        return alignedImage
    }

    func extractEmbedding(from image: CIImage) async throws -> [Float] {
        let alignedImage = try alignFace(for: image)
        return try await runCoreMLModel(on: alignedImage)
    }
    
    func extractEmbeddingAndImage(from image: CIImage) async throws -> (embedding: [Float], alignedImage: CIImage) {
        let alignedImage = try alignFace(for: image)
        let embedding = try await runCoreMLModel(on: alignedImage)
        return (embedding, alignedImage)
    }
    
    private func runCoreMLModel(on image: CIImage) async throws -> [Float] {
        let pixelBuffer = try createPixelBuffer(from: image)
        
        let inputName = mlModel.modelDescription.inputDescriptionsByName.first!.key
        let input = try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(pixelBuffer: pixelBuffer)])
        
        let output = try await mlModel.prediction(from: input)

        let outputName = mlModel.modelDescription.outputDescriptionsByName.first!.key
        guard let multiArray = output.featureValue(for: outputName)?.multiArrayValue else {
            throw FacialRecognitionError.processingFailed
        }

        var embedding: [Float] = []
        let length = multiArray.count

        for i in 0..<length {
            let value = multiArray[i].floatValue
            embedding.append(value)
        }

        guard embedding.count == embeddingSize else {
            print("⚠️ Embedding size mismatch: got \(embedding.count), expected \(embeddingSize)")
            throw FacialRecognitionError.processingFailed
        }

        return normalize(embedding)
    }

    private func resizeImage(_ image: CIImage, to size: CGSize) -> CIImage {
        let scaleX = size.width / image.extent.width
        let scaleY = size.height / image.extent.height

        return image.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
    }

    private func createPixelBuffer(from image: CIImage) throws -> CVPixelBuffer {
        let width = Int(modelInputSize.width)
        let height = Int(modelInputSize.height)

        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA
        ] as CFDictionary

        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA,
            attrs,
            &pixelBuffer
        )

        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            throw FacialRecognitionError.processingFailed
        }

        // Disable Color Management: Use "null" color space to prevent sRGB gamma application
        // This ensures pixel values are copied raw, matching Python's OpenCV/PIL behavior.
        let context = CIContext(options: [
            .workingColorSpace: NSNull(), 
            .outputColorSpace: NSNull()
        ])
        context.render(image, to: buffer)

        // DEBUG: Save Pixel Buffer to verify EXACT model input
        let debugPath = FileManager.default.temporaryDirectory.appendingPathComponent("debug_model_input.jpg")
        if let cgImage = context.createCGImage(image, from: image.extent) {
             let uiImage = UIImage(cgImage: cgImage)
             try? uiImage.jpegData(compressionQuality: 1.0)?.write(to: debugPath)
             print("DEBUG: Saved model input to \(debugPath.path)")
        }
        
        return buffer
    }

    // MARK: Private - Math Utilities

    private func calculateCosineSimilarity(_ vec1: [Float], _ vec2: [Float]) -> Float {

        return MathUtilities.calculateCosineSimilarity(vec1, vec2)
    }

    private func normalize(_ vector: [Float]) -> [Float] {
        return MathUtilities.normalize(vector)
    }
    
}
