import CoreGraphics
import CoreML
import UIKit
import Vision

/// Blood coverage: the same color rule as calibre/components/blood.py.
enum BloodDetector {
    static let width = 480, height = 270
    static let minSaturation = 0.75  // blood is deep, saturated red
    static let maxBrightness = 0.75  // but not glare or bright pink tissue
    static let minBrightnessInView = 0.12  // darker than this is the black border outside the endoscope

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

        var bloodMask = [UInt8](repeating: 0, count: width * height * 4)
        var pixelsInView = 0, bloodPixels = 0
        for pixel in stride(from: 0, to: pixels.count, by: 4) {
            let red = Double(pixels[pixel]) / 255
            let green = Double(pixels[pixel + 1]) / 255
            let blue = Double(pixels[pixel + 2]) / 255
            let brightest = max(red, green, blue), darkest = min(red, green, blue)
            let brightness = brightest
            guard brightness > minBrightnessInView else { continue }
            pixelsInView += 1
            let saturation = brightest == 0 ? 0 : (brightest - darkest) / brightest
            guard saturation > minSaturation, brightness < maxBrightness else { continue }
            // Hue from 0 to 1. Red sits at both ends of the hue circle.
            let colorRange = brightest - darkest
            var hue: Double
            if colorRange == 0 { hue = 0 }
            else if brightest == red { hue = ((green - blue) / colorRange).truncatingRemainder(dividingBy: 6) / 6 }
            else if brightest == green { hue = ((blue - red) / colorRange + 2) / 6 }
            else { hue = ((red - green) / colorRange + 4) / 6 }
            if hue < 0 { hue += 1 }
            let isRed = hue < 0.04 || hue > 0.94
            if isRed {
                bloodPixels += 1
                bloodMask[pixel + 1] = 55; bloodMask[pixel + 3] = 55  // faint green tint (premultiplied)
            }
        }

        let maskImage = bloodMask.withUnsafeMutableBytes { buffer -> CGImage? in
            CGContext(data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8,
                      bytesPerRow: width * 4, space: space, bitmapInfo: info)?.makeImage()
        }
        let fraction = pixelsInView == 0 ? 0 : Double(bloodPixels) / Double(pixelsInView)
        return (fraction, maskImage)
    }
}

/// The YOLO11 tool detector (trained by calibre/tool_detector/train_tool_detector.py),
/// running on the iPhone through Core ML and Vision.
final class ToolDetector {
    static let shared = ToolDetector()
    private let model: VNCoreMLModel?

    var isAvailable: Bool { model != nil }

    private init() {
        if let url = Bundle.main.url(forResource: "CalibreToolDetector", withExtension: "mlmodelc"),
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
