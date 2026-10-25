import SwiftUI

@main
struct CalibreApp: App {
    var body: some Scene {
        WindowGroup {
            HomeView()
                .preferredColorScheme(.dark)
                .tint(.white)
        }
    }
}
