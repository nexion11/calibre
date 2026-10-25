import PhotosUI
import SwiftUI

/// What the scope screen should open with.
struct ScopeTarget: Identifiable {
    let id = UUID()
    let mode: SessionEngine.Mode
}

enum HomeColors {
    static let ink = Color(red: 0.08, green: 0.15, blue: 0.28)
    static let soft = Color(red: 0.33, green: 0.42, blue: 0.55)
    static let blue = Color(red: 0.23, green: 0.48, blue: 0.88)
    static let lightBlue = Color(red: 0.66, green: 0.78, blue: 0.96)
    static let background = LinearGradient(colors: [Color(red: 0.90, green: 0.94, blue: 0.99),
                                                    Color(red: 0.97, green: 0.98, blue: 1.0)],
                                           startPoint: .top, endPoint: .bottom)
}

struct HomeView: View {
    @State private var target: ScopeTarget?
    @State private var pickedItem: PhotosPickerItem?
    @State private var loadingVideo = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Spacer()

            HStack(spacing: 14) {
                Text("C.A.L.I.B.R.E")
                    .font(.system(size: 40, weight: .light))
                    .foregroundStyle(HomeColors.ink)
                CalibreLogo().frame(width: 52, height: 52)
            }

            fullName
                .padding(.top, 14)

            Spacer()

            VStack(spacing: 12) {
                Button { target = ScopeTarget(mode: .camera) } label: {
                    HomeButton(icon: "camera", title: "Live camera")
                }
                PhotosPicker(selection: $pickedItem, matching: .videos) {
                    HomeButton(icon: "film", title: loadingVideo ? "Loading…" : "Open video")
                }
            }
            .buttonStyle(.plain)

            Spacer()
        }
        .padding(28)
        .background(HomeColors.background.ignoresSafeArea())
        .onChange(of: pickedItem) { _, item in
            guard let item else { return }
            loadingVideo = true
            Task {
                if let video = try? await item.loadTransferable(type: PickedVideo.self) {
                    target = ScopeTarget(mode: .file(video.url, sample: nil))
                }
                loadingVideo = false
                pickedItem = nil
            }
        }
        .fullScreenCover(item: $target) { target in
            ScopeView(engine: SessionEngine(mode: target.mode))
                .preferredColorScheme(.dark)
        }
    }

    /// "Carotid Artery Laceration Intelligent Bleed Response Evaluator", first letters bold.
    private var fullName: some View {
        let words = ["Carotid", "Artery", "Laceration", "Intelligent", "Bleed", "Response", "Evaluator"]
        let lines = [words[0...2], words[3...4], words[5...6]]
        return VStack(alignment: .leading, spacing: 2) {
            ForEach(0..<lines.count, id: \.self) { i in
                lines[i].reduce(Text("")) { line, word in
                    line + Text(word.prefix(1)).bold() + Text(word.dropFirst() + " ")
                }
            }
        }
        .font(.system(size: 22))
        .foregroundStyle(HomeColors.ink)
    }
}

private struct HomeButton: View {
    let icon: String
    let title: String

    var body: some View {
        HStack(spacing: 14) {
            Image(systemName: icon)
                .font(.system(size: 19))
                .foregroundStyle(HomeColors.blue)
                .frame(width: 26)
            Text(title)
                .font(.system(size: 18, weight: .semibold))
                .foregroundStyle(HomeColors.ink)
            Spacer()
            Image(systemName: "chevron.right")
                .font(.footnote)
                .foregroundStyle(HomeColors.soft)
        }
        .padding(.vertical, 18)
        .padding(.horizontal, 18)
        .background(Color.white, in: RoundedRectangle(cornerRadius: 14))
        .overlay(RoundedRectangle(cornerRadius: 14).stroke(HomeColors.lightBlue.opacity(0.6), lineWidth: 1))
    }
}

/// The CALIBRE logo: a blue crescent with a lighter crescent inside it.
struct CalibreLogo: View {
    var body: some View {
        Canvas { context, size in
            // A crescent is a circle with a smaller circle cut out of its right side.
            func crescent(_ outer: CGRect, cut: CGRect, color: Color) {
                context.drawLayer { layer in
                    layer.fill(Path(ellipseIn: outer), with: .color(color))
                    layer.blendMode = .destinationOut
                    layer.fill(Path(ellipseIn: cut), with: .color(.black))
                }
            }
            let s = min(size.width, size.height)
            crescent(CGRect(x: 0, y: 0, width: s, height: s),
                     cut: CGRect(x: s * 0.38, y: s * 0.14, width: s * 0.72, height: s * 0.72), color: HomeColors.blue)
            crescent(CGRect(x: s * 0.3, y: s * 0.22, width: s * 0.56, height: s * 0.56),
                     cut: CGRect(x: s * 0.52, y: s * 0.3, width: s * 0.4, height: s * 0.4), color: HomeColors.lightBlue)
        }
    }
}
