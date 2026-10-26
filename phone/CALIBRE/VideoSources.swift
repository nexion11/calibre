import AVFoundation
import CoreImage

/// Anything that shows smooth video and hands CALIBRE one frame per second.
protocol FrameSource: AnyObject {
    var onFrame: ((CGImage) -> Void)? { get set }
    var onFinish: (() -> Void)? { get set }
    func start()
    func stop()
}

/// Back camera aimed at a surgical monitor. The preview fills the screen;
/// analysis uses only the 16:9 framing guide in the middle, where the monitor should be.
final class CameraSource: NSObject, FrameSource, AVCaptureVideoDataOutputSampleBufferDelegate {
    var onFrame: ((CGImage) -> Void)?
    var onFinish: (() -> Void)?
    let session = AVCaptureSession()
    private let queue = DispatchQueue(label: "calibre.camera")
    private let context = CIContext()
    private var lastFrame = Date.distantPast

    /// Width of the framing guide as a share of the portrait frame width.
    static let guideWidth = 0.92

    func start() {
        queue.async { [self] in
            if session.inputs.isEmpty {
                guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
                      let input = try? AVCaptureDeviceInput(device: device) else { return }
                session.sessionPreset = .hd1920x1080
                session.addInput(input)
                let output = AVCaptureVideoDataOutput()
                output.setSampleBufferDelegate(self, queue: queue)
                session.addOutput(output)
                output.connection(with: .video)?.videoRotationAngle = 90  // portrait, like the preview
            }
            session.startRunning()
        }
    }

    func stop() {
        queue.async { [self] in session.stopRunning() }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard Date().timeIntervalSince(lastFrame) >= 1,
              let buffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        lastFrame = Date()
        let image = CIImage(cvPixelBuffer: buffer)
        let full = image.extent
        let w = full.width * Self.guideWidth, h = w * 9 / 16
        let guide = CGRect(x: full.midX - w / 2, y: full.midY - h / 2, width: w, height: h)
        if let cgImage = context.createCGImage(image, from: guide) {
            onFrame?(cgImage)
        }
    }
}

/// A recorded video file. Plays normally and grabs the frame at every whole second.
final class FileSource: FrameSource {
    var onFrame: ((CGImage) -> Void)?
    var onFinish: (() -> Void)?
    let player: AVPlayer
    private let output = AVPlayerItemVideoOutput(pixelBufferAttributes: nil)
    private let context = CIContext()
    private var timeObserver: Any?
    private var endObserver: NSObjectProtocol?
    private var transform = CGAffineTransform.identity
    private var lastSecond = -1

    init(url: URL) {
        let item = AVPlayerItem(url: url)
        item.add(output)
        player = AVPlayer(playerItem: item)
        player.isMuted = true
    }

    /// Width / height of the video as it should be shown (after rotation).
    func displayAspect() async -> CGFloat {
        guard let track = try? await player.currentItem?.asset.loadTracks(withMediaType: .video).first,
              let (size, preferred) = try? await track.load(.naturalSize, .preferredTransform) else { return 16 / 9 }
        transform = preferred
        let shown = CGRect(origin: .zero, size: size).applying(preferred)
        return abs(shown.width / shown.height)
    }

    func start() {
        guard timeObserver == nil else { return }  // onAppear can fire twice
        timeObserver = player.addPeriodicTimeObserver(forInterval: CMTime(value: 1, timescale: 1),
                                                      queue: .main) { [weak self] time in
            // the observer also fires on play/pause, so only take each whole second once
            guard let self, time.seconds >= 0.5, Int(time.seconds) != lastSecond,
                  let buffer = output.copyPixelBuffer(forItemTime: time, itemTimeForDisplay: nil) else { return }
            var image = CIImage(cvPixelBuffer: buffer).transformed(by: transform)
            image = image.transformed(by: CGAffineTransform(translationX: -image.extent.minX, y: -image.extent.minY))
            lastSecond = Int(time.seconds)
            if let cgImage = context.createCGImage(image, from: image.extent) {
                DispatchQueue.global(qos: .userInitiated).async { self.onFrame?(cgImage) }
            }
        }
        endObserver = NotificationCenter.default.addObserver(forName: .AVPlayerItemDidPlayToEndTime,
                                                             object: player.currentItem, queue: .main) { [weak self] _ in
            self?.onFinish?()
        }
        player.play()
    }

    func stop() {
        player.pause()
        if let timeObserver { player.removeTimeObserver(timeObserver) }
        if let endObserver { NotificationCenter.default.removeObserver(endObserver) }
        timeObserver = nil
        endObserver = nil
    }
}
