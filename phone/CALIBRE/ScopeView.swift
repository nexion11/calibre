import AVFoundation
import SwiftUI

enum Theme {
    static let panel = Color(white: 0.08)
    static let line = Color(white: 0.18)
    static let alert = Color(red: 0.95, green: 0.27, blue: 0.23)
    static let normal = Color(red: 0.45, green: 0.85, blue: 0.55)
    static let caution = Color(red: 0.98, green: 0.72, blue: 0.25)
    static let label = Font.system(size: 13, weight: .medium)
    /// One color per tool, used for its box and its row in the timeline.
    static func tool(_ label: String) -> Color {
        switch label {
        case "suction": return Color(red: 0.40, green: 0.75, blue: 1.0)
        case "grasper": return Color(red: 0.98, green: 0.80, blue: 0.35)
        case "cottonoid": return Color(red: 0.75, green: 0.60, blue: 1.0)
        case "muscle": return Color(red: 1.0, green: 0.55, blue: 0.70)
        case "string": return Color(red: 0.55, green: 0.90, blue: 0.70)
        default: return .white
        }
    }
    static func mono(_ size: CGFloat, _ weight: Font.Weight = .regular) -> Font {
        .system(size: size, weight: weight, design: .monospaced)
    }
}

/// The full-screen monitor: video with tool boxes, and a readout strip below.
struct ScopeView: View {
    @StateObject var engine: SessionEngine
    @Environment(\.dismiss) private var dismiss
    @State private var showBlood = false
    @State private var showModelInfo = false

    var body: some View {
        VStack(spacing: 0) {
            topBar
            video
                .onTapGesture { showBlood.toggle() }
                .overlay(alignment: .bottomTrailing) {
                    Text(showBlood ? "Blood map on" : "Tap for blood map")
                        .font(.caption)
                        .padding(.horizontal, 6).padding(.vertical, 3)
                        .background(Color.black.opacity(0.6), in: Capsule())
                        .padding(8)
                        .allowsHitTesting(false)
                }
            readout
            Spacer(minLength: 0)
        }
        .background(Color.black.ignoresSafeArea())
        .foregroundStyle(.white)
        .sheet(isPresented: $showModelInfo) { ModelInfoSheet() }
        .onAppear { engine.start() }
        .onDisappear { engine.stop() }
        .statusBarHidden()
    }

    // MARK: Top bar

    private var topBar: some View {
        HStack {
            Button { dismiss() } label: {
                Image(systemName: "xmark").font(.system(size: 15, weight: .semibold))
                    .frame(width: 36, height: 36)
            }
            Spacer()
            HStack(spacing: 6) {
                if case .camera = engine.mode {
                    Circle().fill(Theme.alert).frame(width: 7, height: 7)
                    Text("Live")
                } else {
                    Text(engine.isFinished ? "Video ended" : "Playback")
                }
            }
            .font(Theme.label)
            Spacer()
            Button { showModelInfo = true } label: {
                Image(systemName: "info.circle").font(.system(size: 17)).frame(width: 36, height: 36)
            }
        }
        .padding(.horizontal, 12)
        .frame(height: 44)
    }

    // MARK: Video

    @ViewBuilder private var video: some View {
        switch engine.mode {
        case .camera:
            ZStack {
                if let camera = engine.source as? CameraSource {
                    CameraPreview(session: camera.session)
                }
                GeometryReader { geo in
                    let w = geo.size.width * CameraSource.guideWidth, h = w * 9 / 16
                    let guide = CGRect(x: (geo.size.width - w) / 2, y: (geo.size.height - h) / 2, width: w, height: h)
                    FramingGuide(rect: guide)
                    overlays.frame(width: w, height: h).position(x: guide.midX, y: guide.midY)
                }
            }
            .aspectRatio(3 / 4, contentMode: .fit)
            .clipped()
        case .file:
            // Square stage, cropped in so the round endoscope view fills it.
            // Video and overlays share one frame, so the boxes stay lined up.
            GeometryReader { geo in
                let size = fillSize(aspect: engine.aspect, in: geo.size)
                ZStack {
                    if let file = engine.source as? FileSource {
                        PlayerView(player: file.player)
                    }
                    overlays
                }
                .frame(width: size.width, height: size.height)
                .position(x: geo.size.width / 2, y: geo.size.height / 2)
            }
            .aspectRatio(1, contentMode: .fit)
            .clipped()
        }
    }

    /// Size that covers `container` while keeping the video's shape.
    private func fillSize(aspect: CGFloat, in container: CGSize) -> CGSize {
        let containerAspect = container.width / max(container.height, 1)
        return aspect > containerAspect
            ? CGSize(width: container.height * aspect, height: container.height)
            : CGSize(width: container.width, height: container.width / aspect)
    }

    private var overlays: some View {
        GeometryReader { geo in
            if showBlood, let mask = engine.bloodMask {
                Image(uiImage: mask).resizable()
            }
            ForEach(engine.tools) { tool in
                ToolMarker(tool: tool, size: geo.size)
            }
        }
        .allowsHitTesting(false)
    }

    // MARK: Readout

    private var readout: some View {
        VStack(alignment: .leading, spacing: 14) {
            VStack(alignment: .leading, spacing: 6) {
                Text("Bleeding risk").font(Theme.label).foregroundStyle(.secondary)
                Text(engine.tier.map { $0.name.capitalized } ?? "Analyzing…")
                    .font(.system(size: 38, weight: .semibold))
                    .foregroundStyle(tierColor)
                    .contentTransition(.opacity)
                    .animation(.default, value: engine.tier)
                modelOutput
            }

            VStack(alignment: .leading, spacing: 4) {
                Sparkline(values: engine.history, blood: engine.bloodHistory,
                          threshold: engine.model.alertThreshold,
                          length: max(engine.second, engine.model.windowSeconds), alert: engine.isAlert)
                    .frame(height: 48)
                HStack(spacing: 12) {
                    Label("Estimate", systemImage: "line.diagonal")
                    Label("Blood in view", systemImage: "square.fill").foregroundStyle(Theme.alert.opacity(0.6))
                }
                .font(.caption2)
                .foregroundStyle(.secondary)
            }

            HStack(spacing: 0) {
                Metric(label: "Blood in view", value: "\(Int(engine.bloodFraction * 100))%")
                Metric(label: "Tools in view", value: engine.detectorAvailable ? "\(engine.tools.count)" : "—")
            }

            ToolTimeline(timeline: engine.toolTimeline, now: engine.second,
                         seconds: max(engine.second, engine.model.windowSeconds))

            Text(clock(engine.second))
                .font(.system(size: 28, weight: .medium).monospacedDigit())
                .frame(maxWidth: .infinity)
                .padding(.top, 2)
        }
        .padding(.horizontal, 16)
        .padding(.top, 16)
    }

    /// What the model outputs, written out.
    private var modelOutput: some View {
        VStack(alignment: .leading, spacing: 4) {
            if let tier = engine.tier, let estimate = engine.estimateML, let range = engine.range {
                OutputRow(label: "Probability of severe bleed", value: "\(Int((tier.severeRate * 100).rounded()))%")
                OutputRow(label: "Estimated blood loss", value: "\(Int(estimate)) mL")
                OutputRow(label: "Likely range", value: "\(range.low)–\(range.high) mL")
            } else {
                Text("Watching… \(engine.second) of \(SessionEngine.firstCallSecond) seconds")
                    .font(.subheadline).foregroundStyle(.secondary)
            }
        }
    }

    private var tierColor: Color {
        switch engine.tier?.name {
        case "HIGH": return Theme.alert
        case "MODERATE": return Theme.caution
        case "LOW": return Theme.normal
        default: return .gray
        }
    }

    private func clock(_ seconds: Int) -> String {
        String(format: "%02d:%02d", seconds / 60, seconds % 60)
    }
}

// MARK: - Pieces

/// When each tool was on screen: one row per tool, a bar for each stretch it was used.
private struct ToolTimeline: View {
    let timeline: [String: [Int]]
    let now: Int
    let seconds: Int
    private let tools = ["suction", "grasper", "cottonoid", "muscle", "string"]

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            HStack {
                Text("Tool use").font(Theme.label).foregroundStyle(.secondary)
                Spacer()
                Text("first used · total").font(.caption2).foregroundStyle(.tertiary)
            }
            ForEach(tools, id: \.self) { tool in
                let seen = timeline[tool] ?? []
                HStack(spacing: 8) {
                    HStack(spacing: 5) {
                        Circle().fill(Theme.tool(tool)).frame(width: 6, height: 6)
                        Text(tool).font(.caption)
                    }
                    .frame(width: 78, alignment: .leading)
                    GeometryReader { geo in
                        ZStack(alignment: .leading) {
                            Rectangle().fill(Theme.line).frame(height: 1)
                            ForEach(stretches(seen), id: \.start) { run in
                                RoundedRectangle(cornerRadius: 2).fill(Theme.tool(tool))
                                    .frame(width: max(geo.size.width * CGFloat(run.length) / CGFloat(seconds), 2), height: 9)
                                    .offset(x: geo.size.width * CGFloat(run.start - 1) / CGFloat(seconds))
                            }
                        }
                        .frame(height: 9)
                    }
                    .frame(height: 9)
                    Text(seen.first.map { "\(clock($0)) · \(seen.count)s" } ?? "not used")
                        .font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
                        .frame(width: 72, alignment: .trailing)
                }
            }
            // time axis under the bars, with a line for where we are now
            HStack(spacing: 8) {
                Color.clear.frame(width: 78, height: 1)
                GeometryReader { geo in
                    ZStack(alignment: .topLeading) {
                        ForEach(Array(stride(from: 0, through: seconds, by: 15)), id: \.self) { t in
                            Text(clock(t)).font(.system(size: 9).monospacedDigit()).foregroundStyle(.tertiary)
                                .fixedSize()
                                .position(x: geo.size.width * CGFloat(t) / CGFloat(seconds), y: 6)
                        }
                    }
                }
                .frame(height: 12)
                Color.clear.frame(width: 72, height: 1)
            }
        }
    }

    /// Turns [3, 4, 5, 9, 10] into stretches 3–5 and 9–10.
    private func stretches(_ seen: [Int]) -> [(start: Int, length: Int)] {
        var runs: [(start: Int, length: Int)] = []
        for second in seen.sorted() {
            if let last = runs.last, last.start + last.length == second {
                runs[runs.count - 1].length += 1
            } else {
                runs.append((second, 1))
            }
        }
        return runs
    }

    private func clock(_ s: Int) -> String { "\(s / 60):\(String(format: "%02d", s % 60))" }
}

/// The charts made by calibre/components/charts.py and timing_analysis.py.
private struct ModelInfoSheet: View {
    private let charts = [
        ("Estimate vs what really happened",
         "predicted_vs_actual.png",
         "Each dot is one trial. Dots right of the dotted line got an alert. Most red dots (severe bleeds) are on the right."),
        ("What the model weighs",
         "model_weights.png",
         "How much each thing CALIBRE measures pushes the estimate up or down."),
        ("Accuracy over time",
         "live_accuracy.png",
         "Tested on surgeons the model never saw. AUC: 0.5 = coin flip, 1 = perfect."),
    ]

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 28) {
                Text("How the model works").font(.title3.bold())
                ForEach(charts, id: \.1) { title, file, caption in
                    VStack(alignment: .leading, spacing: 8) {
                        Text(title).font(.headline)
                        if let chart = UIImage(named: file) {
                            Image(uiImage: chart).resizable().scaledToFit()
                                .background(Color.white)
                                .clipShape(RoundedRectangle(cornerRadius: 8))
                        }
                        Text(caption).font(.footnote).foregroundStyle(.secondary)
                    }
                }
            }
            .padding(20)
        }
        .presentationDetents([.medium, .large])
    }
}

private struct OutputRow: View {
    let label: String
    let value: String

    var body: some View {
        HStack {
            Text(label).foregroundStyle(.secondary)
            Spacer()
            Text(value).monospacedDigit()
        }
        .font(.subheadline)
    }
}

private struct Metric: View {
    let label: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(label).font(Theme.label).foregroundStyle(.secondary)
            Text(value).font(.system(size: 16, weight: .medium).monospacedDigit())
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

private struct ToolMarker: View {
    let tool: ToolBox
    let size: CGSize

    var body: some View {
        let rect = CGRect(x: tool.x * size.width, y: tool.y * size.height,
                          width: tool.w * size.width, height: tool.h * size.height)
        ZStack(alignment: .topLeading) {
            RoundedRectangle(cornerRadius: 3).stroke(Theme.tool(tool.label), lineWidth: 2)
            Text("\(tool.label) \(Int(tool.confidence * 100))%")
                .font(.system(size: 10, weight: .semibold))
                .padding(.horizontal, 4).padding(.vertical, 1)
                .background(Theme.tool(tool.label), in: RoundedRectangle(cornerRadius: 3))
                .foregroundStyle(.black)
                .offset(y: -16)
        }
        .frame(width: rect.width, height: rect.height)
        .position(x: rect.midX, y: rect.midY)
    }
}

/// Corner brackets showing where to frame the surgical monitor.
private struct FramingGuide: View {
    let rect: CGRect

    var body: some View {
        Path { p in
            let l: CGFloat = 18
            for (corner, dx, dy) in [(CGPoint(x: rect.minX, y: rect.minY), 1.0, 1.0),
                                     (CGPoint(x: rect.maxX, y: rect.minY), -1.0, 1.0),
                                     (CGPoint(x: rect.minX, y: rect.maxY), 1.0, -1.0),
                                     (CGPoint(x: rect.maxX, y: rect.maxY), -1.0, -1.0)] {
                p.move(to: CGPoint(x: corner.x + dx * l, y: corner.y))
                p.addLine(to: corner)
                p.addLine(to: CGPoint(x: corner.x, y: corner.y + dy * l))
            }
        }
        .stroke(Color.white.opacity(0.8), lineWidth: 2)
    }
}

/// The estimate over the first 60 seconds, with the alert line dashed.
/// Blood in view is shaded behind it (0–100% top to bottom).
private struct Sparkline: View {
    let values: [Double]
    let blood: [Double]
    let threshold: Double
    let length: Int
    let alert: Bool

    var body: some View {
        GeometryReader { geo in
            let top = max((values.max() ?? 0) * 1.15, threshold * 2)
            let x = { (i: Int) in geo.size.width * CGFloat(i) / CGFloat(max(length - 1, 1)) }
            let y = { (v: Double) in geo.size.height * (1 - CGFloat(v / top)) }
            ZStack {
                Path { p in
                    guard !blood.isEmpty else { return }
                    p.move(to: CGPoint(x: 0, y: geo.size.height))
                    for (i, b) in blood.enumerated() {
                        p.addLine(to: CGPoint(x: x(i), y: geo.size.height * (1 - CGFloat(b))))
                    }
                    p.addLine(to: CGPoint(x: x(blood.count - 1), y: geo.size.height))
                    p.closeSubpath()
                }
                .fill(Theme.alert.opacity(0.18))
                Path { p in
                    p.move(to: CGPoint(x: 0, y: y(threshold)))
                    p.addLine(to: CGPoint(x: geo.size.width, y: y(threshold)))
                }
                .stroke(Color.white.opacity(0.35), style: StrokeStyle(lineWidth: 1, dash: [3, 3]))
                Path { p in
                    for (i, v) in values.enumerated() {
                        i == 0 ? p.move(to: CGPoint(x: x(i), y: y(v))) : p.addLine(to: CGPoint(x: x(i), y: y(v)))
                    }
                }
                .stroke(alert ? Theme.alert : Theme.normal, lineWidth: 1.5)
            }
        }
        .overlay(alignment: .bottom) { Rectangle().fill(Theme.line).frame(height: 1) }
    }
}

// MARK: - Video layers

private struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewUIView {
        let view = PreviewUIView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        return view
    }

    func updateUIView(_ uiView: PreviewUIView, context: Context) {}

    final class PreviewUIView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }
}

private struct PlayerView: UIViewRepresentable {
    let player: AVPlayer

    func makeUIView(context: Context) -> PlayerUIView {
        let view = PlayerUIView()
        view.playerLayer.player = player
        view.playerLayer.videoGravity = .resizeAspect
        return view
    }

    func updateUIView(_ uiView: PlayerUIView, context: Context) {}

    final class PlayerUIView: UIView {
        override class var layerClass: AnyClass { AVPlayerLayer.self }
        var playerLayer: AVPlayerLayer { layer as! AVPlayerLayer }
    }
}
