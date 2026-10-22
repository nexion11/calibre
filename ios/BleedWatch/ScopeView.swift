import AVFoundation
import SwiftUI

enum Theme {
    static let panel = Color(white: 0.08)
    static let line = Color(white: 0.18)
    static let alert = Color(red: 0.95, green: 0.27, blue: 0.23)
    static let normal = Color(red: 0.45, green: 0.85, blue: 0.55)
    static let label = Font.system(size: 11, weight: .medium).width(.expanded)
    static func mono(_ size: CGFloat, _ weight: Font.Weight = .regular) -> Font {
        .system(size: size, weight: weight, design: .monospaced)
    }
}

/// The full-screen monitor: video with tool boxes, and a readout strip below.
struct ScopeView: View {
    @StateObject var engine: SessionEngine
    @Environment(\.dismiss) private var dismiss
    @State private var showBlood = false

    var body: some View {
        VStack(spacing: 0) {
            topBar
            video
                .onTapGesture { showBlood.toggle() }
                .overlay(alignment: .bottomTrailing) {
                    Text(showBlood ? "BLOOD MAP ON" : "TAP FOR BLOOD MAP")
                        .font(Theme.label)
                        .padding(.horizontal, 6).padding(.vertical, 3)
                        .background(Color.black.opacity(0.6))
                        .padding(8)
                        .allowsHitTesting(false)
                }
            readout
            Spacer(minLength: 0)
        }
        .background(Color.black.ignoresSafeArea())
        .foregroundStyle(.white)
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
                    Text("LIVE")
                } else {
                    Text(engine.isFinished ? "END OF VIDEO" : "PLAYBACK")
                }
            }
            .font(Theme.label)
            Spacer()
            Text(clock(engine.second)).font(Theme.mono(14)).frame(width: 60, alignment: .trailing)
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
            HStack(alignment: .lastTextBaseline) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("EST. BLOOD LOSS").font(Theme.label).foregroundStyle(.secondary)
                    Text(engine.estimateML.map { "\(Int($0).formatted()) mL" } ?? "– – –")
                        .font(Theme.mono(40, .medium))
                        .contentTransition(.numericText())
                        .animation(.default, value: engine.estimateML)
                }
                Spacer()
                status
            }

            Sparkline(values: engine.history, threshold: engine.model.alertThreshold,
                      length: engine.model.windowSeconds, alert: engine.isAlert)
                .frame(height: 44)

            HStack(spacing: 0) {
                Metric(label: "BLOOD", value: "\(Int(engine.bloodFraction * 100))%")
                Metric(label: "TOOLS", value: engine.detectorAvailable ? "\(engine.tools.count)" : "—")
                Metric(label: "WINDOW", value: engine.isLocked ? "LOCKED" : "\(engine.second)/\(engine.model.windowSeconds)s")
            }

            if let reason = engine.reason {
                HStack(spacing: 8) {
                    Rectangle().fill(reason.raisesRisk ? Theme.alert : Theme.normal).frame(width: 2)
                    Text(reason.text).font(.subheadline)
                }
                .fixedSize(horizontal: false, vertical: true)
            }

            if engine.isFinished, let sample = engine.sample {
                VStack(alignment: .leading, spacing: 4) {
                    Text("RECORDED OUTCOME").font(Theme.label).foregroundStyle(.secondary)
                    Text("\(sample.actualBloodLoss.formatted()) mL lost · bleeding \(sample.controlled ? "controlled" : "not controlled") in 5 min")
                        .font(.subheadline)
                }
                .padding(.top, 4)
            }

            if !engine.detectorAvailable {
                Text("Tool detector not installed; estimate uses blood only.")
                    .font(.caption).foregroundStyle(.tertiary)
            }
        }
        .padding(.horizontal, 16)
        .padding(.top, 16)
    }

    private var status: some View {
        let text = engine.estimateML == nil ? "ANALYZING" : engine.isAlert ? "HIGH RISK" : "STABLE"
        let color = engine.estimateML == nil ? Color.gray : engine.isAlert ? Theme.alert : Theme.normal
        return Text(text)
            .font(Theme.mono(13, .semibold))
            .padding(.horizontal, 10).padding(.vertical, 6)
            .overlay(Rectangle().stroke(color, lineWidth: 1.5))
            .foregroundStyle(color)
    }

    private func clock(_ seconds: Int) -> String {
        String(format: "%02d:%02d", seconds / 60, seconds % 60)
    }
}

// MARK: - Pieces

private struct Metric: View {
    let label: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(label).font(Theme.label).foregroundStyle(.secondary)
            Text(value).font(Theme.mono(15))
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
            Rectangle().stroke(Color.white.opacity(0.9), lineWidth: 1.5)
            Text("\(tool.label.uppercased()) \(Int(tool.confidence * 100))")
                .font(Theme.mono(9, .semibold))
                .padding(.horizontal, 3).padding(.vertical, 1)
                .background(Color.white)
                .foregroundStyle(.black)
                .offset(y: -14)
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
private struct Sparkline: View {
    let values: [Double]
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
