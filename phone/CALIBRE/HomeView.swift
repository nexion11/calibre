import PhotosUI
import SwiftUI

/// What the scope screen should open with.
struct ScopeTarget: Identifiable {
    let id = UUID()
    let mode: SessionEngine.Mode
}

struct HomeView: View {
    @State private var target: ScopeTarget?
    @State private var pickedItem: PhotosPickerItem?
    @State private var loadingVideo = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Spacer()

            Text("CALIBRE")
                .font(.system(size: 34, weight: .semibold))
            Text("AI coach for carotid-injury\nsimulation training")
                .font(.callout)
                .foregroundStyle(.secondary)
                .padding(.top, 6)

            Spacer()

            VStack(spacing: 1) {
                HomeButton(icon: "camera", title: "Live camera", detail: "Aim at the surgical monitor") {
                    target = ScopeTarget(mode: .camera)
                }
                PhotosPicker(selection: $pickedItem, matching: .videos) {
                    HomeRow(icon: "film", title: loadingVideo ? "Loading…" : "Open video",
                            detail: "From your photo library")
                }
                .buttonStyle(.plain)
            }
            .background(Theme.line)

            VStack(alignment: .leading, spacing: 10) {
                Text("SAMPLE CASES").font(Theme.label).foregroundStyle(.secondary)
                ForEach(SampleCase.all) { sample in
                    Button {
                        if let url = sample.url { target = ScopeTarget(mode: .file(url, sample: sample)) }
                    } label: {
                        HStack {
                            Text(sample.title)
                            Spacer()
                            Text(sample.id).font(Theme.mono(13)).foregroundStyle(.secondary)
                        }
                    }
                    .foregroundStyle(.primary)
                }
            }
            .padding(.top, 32)

            Spacer()

            Text("Training tool. Built on the SOCAL cadaver-simulation dataset; not for clinical use.")
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
        .padding(24)
        .background(Color.black.ignoresSafeArea())
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
        }
    }
}

private struct HomeButton: View {
    let icon: String
    let title: String
    let detail: String
    let action: () -> Void

    var body: some View {
        Button(action: action) { HomeRow(icon: icon, title: title, detail: detail) }
            .buttonStyle(.plain)
    }
}

private struct HomeRow: View {
    let icon: String
    let title: String
    let detail: String

    var body: some View {
        HStack(spacing: 16) {
            Image(systemName: icon)
                .font(.system(size: 20, weight: .regular))
                .frame(width: 28)
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(.headline)
                Text(detail).font(.footnote).foregroundStyle(.secondary)
            }
            Spacer()
            Image(systemName: "chevron.right").font(.footnote).foregroundStyle(.tertiary)
        }
        .padding(.vertical, 18)
        .padding(.horizontal, 16)
        .background(Theme.panel)
        .foregroundStyle(.primary)
    }
}
