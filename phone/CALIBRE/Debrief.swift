import SwiftUI

/// One real good run from SOCAL to compare against (made by calibre/components/debrief.py).
struct DebriefReference: Decodable {
    struct Habit: Decodable {
        let best_high: Double
    }

    struct GoodRun: Decodable {
        let trial_id: String
        let blood_loss: Double
        let length: Int
        let first_used: [String: Int]
        let habits: [String: Double]
    }

    let habits: [String: Habit]
    let good_run: GoodRun

    static func load() -> DebriefReference? {
        guard let url = Bundle.main.url(forResource: "debrief_reference", withExtension: "json"),
              let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(DebriefReference.self, from: data)
    }
}

private func clock(_ s: Int) -> String { "\(s / 60):\(String(format: "%02d", s % 60))" }

extension SessionEngine {
    /// Rows of (what, you, good run).
    func debrief(_ reference: DebriefReference) -> [(String, String, String)] {
        let good = reference.good_run
        func first(_ tool: String) -> Int? { toolTimeline[tool]?.min() }
        // if the run ended before good runs usually use it, it's "not yet" instead of "not used"
        func timing(_ tool: String, _ key: String? = nil) -> String {
            if let s = first(tool) { return clock(s) }
            if let key, Double(second) < (reference.habits[key]?.best_high ?? 0) { return "not yet" }
            return "not used"
        }
        func goodTiming(_ tool: String) -> String { good.first_used[tool].map(clock) ?? "–" }

        let firstMinute = min(second, 60)
        let cottonoidShare = firstMinute == 0 ? 0 :
            Double((toolTimeline["cottonoid"] ?? []).filter { $0 <= 60 }.count) / Double(firstMinute)
        let toolsAtOnce = toolCounts.isEmpty ? 0 : Double(toolCounts.reduce(0, +)) / Double(toolCounts.count)

        return [
            ("Suction in", timing("suction"), goodTiming("suction")),
            ("Grasper in", timing("grasper"), goodTiming("grasper")),
            ("Cottonoid in", timing("cottonoid", "cottonoid_first"), goodTiming("cottonoid")),
            ("Muscle patch in", timing("muscle", "muscle_first"), goodTiming("muscle")),
            ("Cottonoid held, 1st min", "\(Int((cottonoidShare * 100).rounded()))%",
             "\(Int(((good.habits["cottonoid_share_60"] ?? 0) * 100).rounded()))%"),
            ("Tools at once", String(format: "%.1f", toolsAtOnce),
             String(format: "%.1f", good.habits["tools_per_frame"] ?? 0)),
            ("Blood loss", estimateML.map { "\(Int($0)) mL" } ?? "–", "\(Int(good.blood_loss)) mL"),
            ("Run length", clock(second), clock(good.length)),
        ]
    }

}

/// Full page shown when a run ends: your run, then your run next to a good one.
struct DebriefView: View {
    @ObservedObject var engine: SessionEngine
    var onDone: () -> Void
    private let reference = DebriefReference.load()

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 28) {
                HStack(alignment: .firstTextBaseline) {
                    Text("Debrief").font(.largeTitle.bold())
                    Spacer()
                    Button("Done", action: onDone).foregroundStyle(HomeColors.blue)
                }

                VStack(alignment: .leading, spacing: 14) {
                    Text("Your run").font(.title2.bold())
                    stat("Bleeding risk", engine.tier?.name.capitalized ?? "–", color: tierColor)
                    stat("Blood loss (est.)", engine.estimateML.map { "\(Int($0)) mL" } ?? "–")
                    stat("Chance of severe bleed", engine.tier.map { "\(Int(($0.severeRate * 100).rounded()))%" } ?? "–")
                    stat("Length", clock(engine.second))
                }

                if let reference {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Analysis").font(.title2.bold())
                        Text("(comparing your run with a good run from SOCAL)")
                            .font(.callout).foregroundStyle(HomeColors.soft)
                    }

                    VStack(spacing: 0) {
                        HStack {
                            Spacer()
                            Text("You").frame(width: 80, alignment: .trailing)
                            Text("Good run").frame(width: 80, alignment: .trailing)
                        }
                        .font(.footnote.weight(.semibold))
                        .foregroundStyle(HomeColors.soft)
                        .padding(.bottom, 6)

                        ForEach(engine.debrief(reference), id: \.0) { what, you, good in
                            Divider()
                            HStack {
                                Text(what)
                                Spacer()
                                Text(you).fontWeight(.semibold).frame(width: 80, alignment: .trailing)
                                Text(good).foregroundStyle(HomeColors.soft).frame(width: 80, alignment: .trailing)
                            }
                            .font(.system(size: 17).monospacedDigit())
                            .padding(.vertical, 11)
                        }
                    }

                    Text("Good run: SOCAL trial \(reference.good_run.trial_id)")
                        .font(.footnote)
                        .foregroundStyle(HomeColors.soft)
                }
            }
            .padding(24)
        }
        .background(HomeColors.background.ignoresSafeArea())
        .foregroundStyle(HomeColors.ink)
    }

    private func stat(_ label: String, _ value: String, color: Color = HomeColors.ink) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label).font(.title3).foregroundStyle(HomeColors.soft)
            Spacer()
            Text(value).font(.system(size: 26, weight: .semibold).monospacedDigit()).foregroundStyle(color)
        }
    }

    private var tierColor: Color {
        switch engine.tier?.name {
        case "HIGH": return Theme.alert
        case "MODERATE": return Theme.caution
        case "LOW": return Theme.normal
        default: return HomeColors.ink
        }
    }
}
