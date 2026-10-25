import SwiftUI

/// Runs one session: takes one frame per second from the camera or a video,
/// measures blood, finds tools, and updates the blood-loss estimate.
@MainActor
final class SessionEngine: ObservableObject {
    enum Mode {
        case camera
        case file(URL, sample: SampleCase?)
    }

    @Published var tools: [ToolBox] = []
    @Published var bloodMask: UIImage?
    @Published var bloodFraction = 0.0
    @Published var second = 0
    @Published var estimateML: Double?
    @Published var history: [Double] = []
    /// Share of the view that was blood, every second.
    @Published var bloodHistory: [Double] = []
    @Published var reason: Explanation?
    @Published var isFinished = false
    /// Which seconds each tool was seen in, for the tool timeline.
    @Published var toolTimeline: [String: [Int]] = [:]
    @Published var aspect: CGFloat = 16 / 9

    let mode: Mode
    let model = RiskModel.load()
    let source: FrameSource
    private var observations: [Observation] = []

    var isAlert: Bool { hasCall && (estimateML ?? 0) > model.alertThreshold }
    /// No risk call before 15 s: earlier than that the estimate is mostly noise (see live_analysis.py).
    static let firstCallSecond = 15
    var hasCall: Bool { second >= Self.firstCallSecond && estimateML != nil }
    var tier: RiskModel.Tier? { hasCall ? estimateML.map(model.tier(for:)) : nil }
    /// The middle-50% range, rounded to the nearest 10 mL so it doesn't look more precise than it is.
    var range: (low: Int, high: Int)? {
        guard hasCall else { return nil }
        return estimateML.map { (Int(($0 * model.rangeLow / 10).rounded()) * 10, Int(($0 * model.rangeHigh / 10).rounded()) * 10) }
    }
    var isLocked: Bool { second >= model.windowSeconds }
    var detectorAvailable: Bool { ToolDetector.shared.isAvailable }
    var sample: SampleCase? {
        if case .file(_, let sample) = mode { return sample }
        return nil
    }

    init(mode: Mode) {
        self.mode = mode
        switch mode {
        case .camera: source = CameraSource()
        case .file(let url, _): source = FileSource(url: url)
        }
        source.onFrame = { [weak self] image in
            // Runs off the main thread: the heavy pixel work happens here.
            let blood = BloodDetector.analyze(image)
            let tools = ToolDetector.shared.detect(image)
            Task { @MainActor in self?.record(blood: blood, tools: tools) }
        }
        source.onFinish = { [weak self] in
            Task { @MainActor in self?.isFinished = true }
        }
    }

    func start() {
        if let file = source as? FileSource {
            Task {
                aspect = await file.displayAspect()
                file.start()
            }
        } else {
            source.start()
        }
    }

    func stop() { source.stop() }

    private func record(blood: (fraction: Double, mask: CGImage?), tools: [ToolBox]) {
        second += 1
        self.tools = tools
        for label in Set(tools.map(\.label)) { toolTimeline[label, default: []].append(second) }
        bloodFraction = blood.fraction
        bloodHistory.append(blood.fraction)
        bloodMask = blood.mask.map { UIImage(cgImage: $0) }

        guard second <= model.windowSeconds else { return }  // estimate locks at 60 s
        observations.append(Observation(second: second, bloodFraction: blood.fraction, tools: tools))
        let values = model.featureValues(observations, toolsAvailable: detectorAvailable)
        let result = model.predict(values)
        estimateML = result.mL
        history.append(result.mL)
        reason = model.explain(values, result.contributions).first
    }
}
