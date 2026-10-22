import Foundation

/// The ridge regression trained in src/train.py, as plain numbers:
/// predicted mL = exp(intercept + sum(weight * (feature - mean) / scale)).
struct RiskModel: Decodable {
    let features: [String]
    let means: [Double]
    let scales: [Double]
    let weights: [Double]
    let intercept: Double
    let alertThreshold: Double
    let windowSeconds: Int

    static func load() -> RiskModel {
        let url = Bundle.main.url(forResource: "risk_model", withExtension: "json")!
        return try! JSONDecoder().decode(RiskModel.self, from: Data(contentsOf: url))
    }

    /// Same features as src/features.py, computed from what has been seen so far.
    /// If no tool detector is installed, tool features are left out ("assume average").
    func featureValues(_ observations: [Observation], toolsAvailable: Bool = true) -> [String: Double] {
        let window = Array(observations.prefix(windowSeconds))
        guard !window.isEmpty else { return [:] }
        let n = Double(window.count)
        let blood = window.map(\.bloodFraction)
        let lastSecond = window.last!.second
        let late = window.filter { $0.second > lastSecond - 15 }.map(\.bloodFraction)

        var values = [
            "blood_mean": blood.reduce(0, +) / n,
            "blood_max": blood.max() ?? 0,
            "blood_late": late.isEmpty ? 0 : late.reduce(0, +) / Double(late.count),
        ]
        if toolsAvailable {
            values["suction_frac"] = Double(window.filter { $0.has("suction") }.count) / n
            values["cottonoid_frac"] = Double(window.filter { $0.has("cottonoid") }.count) / n
            values["tools_per_frame"] = Double(window.map(\.tools.count).reduce(0, +)) / n
        }
        // A slope through only a few seconds is mostly noise, so wait for 20 s of data.
        // Until then the feature is left out, which means "assume average".
        if window.count >= 20 {
            values["blood_trend"] = slope(x: window.map { Double($0.second) }, y: blood)
        }
        return values
    }

    /// Predicted blood loss in mL, plus each feature's push up (+) or down (−).
    func predict(_ values: [String: Double]) -> (mL: Double, contributions: [String: Double]) {
        var total = intercept
        var contributions: [String: Double] = [:]
        for (i, name) in features.enumerated() {
            // Clamp to ±3 standard deviations: the model never saw values beyond that,
            // so it shouldn't extrapolate wildly from an unusual first few seconds.
            let z = min(max(((values[name] ?? means[i]) - means[i]) / scales[i], -3), 3)
            contributions[name] = weights[i] * z
            total += weights[i] * z
        }
        return (exp(total), contributions)
    }

    /// Plain-English reasons behind the prediction, biggest first.
    func explain(_ values: [String: Double], _ contributions: [String: Double]) -> [Explanation] {
        let percent = { (name: String) in Int(((values[name] ?? 0) * 100).rounded()) }
        let phrases: [String: (up: String, down: String)] = [
            "cottonoid_frac": ("Cottonoid in use \(percent("cottonoid_frac"))% of the time",
                               "Cottonoid in use \(percent("cottonoid_frac"))% of the time"),
            "blood_late": ("Field \(percent("blood_late"))% obscured by blood",
                           "Field clearing, \(percent("blood_late"))% blood"),
            "blood_mean": ("Blood averaging \(percent("blood_mean"))% of the field",
                           "Blood averaging \(percent("blood_mean"))% of the field"),
            "tools_per_frame": ("Few instruments in the field", "Multiple instruments in use"),
            "suction_frac": ("Suction visible \(percent("suction_frac"))% of the time",
                             "Suction visible \(percent("suction_frac"))% of the time"),
        ]
        // blood_max and blood_trend overlap with blood_mean (see train.py), so their
        // individual signs aren't meaningful to a surgeon. We leave them out of the reasons.
        return contributions
            .filter { phrases[$0.key] != nil && abs($0.value) > 0.05 }
            .sorted { abs($0.value) > abs($1.value) }
            .prefix(3)
            .compactMap { name, value in
                guard let phrase = phrases[name] else { return nil }
                return Explanation(text: value > 0 ? phrase.up : phrase.down, raisesRisk: value > 0)
            }
    }

    private func slope(x: [Double], y: [Double]) -> Double {
        guard x.count > 1 else { return 0 }
        let mx = x.reduce(0, +) / Double(x.count)
        let my = y.reduce(0, +) / Double(y.count)
        let num = zip(x, y).map { ($0 - mx) * ($1 - my) }.reduce(0, +)
        let den = x.map { ($0 - mx) * ($0 - mx) }.reduce(0, +)
        return den == 0 ? 0 : num / den
    }
}
