import CoreTransferable
import Foundation
import UniformTypeIdentifiers

/// One surgical tool in a frame. Coordinates are 0–1, origin at the top left.
struct ToolBox: Identifiable {
    let id = UUID()
    let label: String
    let confidence: Float
    let x: Double
    let y: Double
    let w: Double
    let h: Double
}

/// What CALIBRE saw in one second of video.
struct Observation {
    let second: Int
    let bloodFraction: Double
    let tools: [ToolBox]

    func has(_ label: String) -> Bool { tools.contains { $0.label == label } }
}

struct Explanation {
    let text: String
    let raisesRisk: Bool
}

/// Recorded SOCAL trials bundled with the app. Their surgeons were left out of
/// the app model's training, so predictions on them are honest.
struct SampleCase: Identifiable {
    let id: String
    let title: String
    let actualBloodLoss: Int
    let controlled: Bool

    var url: URL? { Bundle.main.url(forResource: "sample_\(id)", withExtension: "mp4") }

    static let all = [
        SampleCase(id: "S814T1", title: "Severe bleed", actualBloodLoss: 957, controlled: false),
        SampleCase(id: "S103T2", title: "Controlled bleed", actualBloodLoss: 50, controlled: true),
        SampleCase(id: "S807T1", title: "Missed case", actualBloodLoss: 1021, controlled: false),
    ]
}

/// A video picked from Photos, copied somewhere the app can read it.
struct PickedVideo: Transferable {
    let url: URL

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { SentTransferredFile($0.url) } importing: { received in
            let copy = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString)
                .appendingPathExtension(received.file.pathExtension)
            try FileManager.default.copyItem(at: received.file, to: copy)
            return PickedVideo(url: copy)
        }
    }
}
