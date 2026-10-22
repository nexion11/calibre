import CoreGraphics
import CoreML
import UIKit
import Vision

/// Blood coverage: the same color rule as src/blood_coverage.py.
enum BloodDetector {
    static let width = 480, height = 270
    static let saturationMin = 0.75, brightnessMax = 0.75

    /// Returns the share of the endoscope view that looks like blood,
    /// plus a green mask image to draw over the video.
    static func analyze(_ image: CGImage) -> (fraction: Double, mask: CGImage?) {
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        let space = CGColorSpaceCreateDeviceRGB()
        let info = CGImageAlphaInfo.premultipliedLast.rawValue
        guard let context = CGContext(data: &pixels, width: width, height: height, bitsPerComponent: 8,
                                      bytesPerRow: width * 4, space: space, bitmapInfo: info) else {
            return (0, nil)
        }
        context.interpolationQuality = .medium
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        var mask = [UInt8](repeating: 0, count: width * height * 4)
        var inView = 0, blood = 0
        for i in stride(from: 0, to: pixels.count, by: 4) {
            let r = Double(pixels[i]) / 255, g = Double(pixels[i + 1]) / 255, b = Double(pixels[i + 2]) / 255
            let maxC = max(r, g, b), minC = min(r, g, b)
            let v = maxC
            guard v > 0.12 else { continue }  // black border outside the endoscope circle
            inView += 1
            let s = maxC == 0 ? 0 : (maxC - minC) / maxC
            guard s > saturationMin, v < brightnessMax else { continue }
            // Hue 0–1. Red sits at both ends of the hue circle.
            let delta = maxC - minC
            var h: Double
            if delta == 0 { h = 0 }
            else if maxC == r { h = ((g - b) / delta).truncatingRemainder(dividingBy: 6) / 6 }
            else if maxC == g { h = ((b - r) / delta + 2) / 6 }
            else { h = ((r - g) / delta + 4) / 6 }
            if h < 0 { h += 1 }
            if h < 0.04 || h > 0.94 {
                blood += 1
                mask[i + 1] = 55; mask[i + 3] = 55  // faint green tint (premultiplied)
            }
        }

        let maskImage = mask.withUnsafeMutableBytes { buffer -> CGImage? in
            CGContext(data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8,
                      bytesPerRow: width * 4, space: space, bitmapInfo: info)?.makeImage()
        }
        return (inView == 0 ? 0 : Double(blood) / Double(inView), maskImage)
    }
}

/// The YOLO tool detector trained in Colab, if its Core ML file is in the app.
/// Add BleedWatchTools.mlpackage to the BleedWatch folder to turn it on.
final class ToolDetector {
    static let shared = ToolDetector()
    private let model: VNCoreMLModel?

    var isAvailable: Bool { model != nil }

    private init() {
        if let url = Bundle.main.url(forResource: "BleedWatchTools", withExtension: "mlmodelc"),
           let mlModel = try? MLModel(contentsOf: url) {
            model = try? VNCoreMLModel(for: mlModel)
        } else {
            model = nil
        }
    }

    func detect(_ image: CGImage) -> [ToolBox] {
        guard let model else { return [] }
        let request = VNCoreMLRequest(model: model)
        request.imageCropAndScaleOption = .scaleFill
        try? VNImageRequestHandler(cgImage: image).perform([request])
        let results = request.results as? [VNRecognizedObjectObservation] ?? []
        return results.filter { $0.confidence > 0.35 }.compactMap { obs in
            guard let label = obs.labels.first?.identifier else { return nil }
            let box = obs.boundingBox  // Vision puts the origin at the bottom left
            return ToolBox(label: label, confidence: obs.confidence,
                           x: box.minX, y: 1 - box.maxY, w: box.width, h: box.height)
        }
    }
}
