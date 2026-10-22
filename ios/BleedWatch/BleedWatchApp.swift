import SwiftUI

@main
struct BleedWatchApp: App {
    var body: some Scene {
        WindowGroup {
            HomeView()
                .preferredColorScheme(.dark)
                .tint(.white)
        }
    }
}
