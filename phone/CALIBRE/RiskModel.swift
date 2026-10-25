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
    /// 80% of held-out trials landed between rangeLow× and rangeHigh× the estimate.
    let rangeLow: Double
    let rangeHigh: Double
    let tiers: [Tier]

    /// A risk level, with how often held-out trials at that level became severe (>500 mL).
    struct Tier: Decodable, Equatable {
        let name: String
        let below: Double?
        let severeRate: Double
    }

    func tier(for estimate: Double) -> Tier {
        tiers.first { $0.below.map { estimate < $0 } ?? true } ?? tiers[tiers.count - 1]
    }

    static func load() -> RiskModel {
        let url = Bundle.main.url(forResource: "risk_model", withExtension: "json")!
        return try! JSONDecoder().decode(RiskModel.self, from: Data(contentsOf: url))
    }

    /// Same features as src/features.py, computed from what has been seen so far.
    /// If no tool detector is installed, tool features are left out ("assume average").
    func featureValues(_ observations: [Observation], toolsAvailable: Bool = true) -> [String: Double] {
        let window = Array(observations.prefix(windowSeconds))
        guard !window.isEmpty else { return [:] }
        let secondsSeen = Double(window.count)
        let blood = window.map(\.bloodFraction)
        let lastSecond = window.last!.second
        let late = window.filter { $0.second > lastSecond - 15 }.map(\.bloodFraction)

        var values = [
            "blood_mean": blood.reduce(0, +) / secondsSeen,
            "blood_max": blood.max() ?? 0,
            "blood_late": late.isEmpty ? 0 : late.reduce(0, +) / Double(late.count),
        ]
        if toolsAvailable {
            values["suction_amt"] = Double(window.filter { $0.has("suction") }.count) / secondsSeen
            values["cottonoid_amt"] = Double(window.filter { $0.has("cottonoid") }.count) / secondsSeen
            values["tools_per_frame"] = Double(window.map(\.tools.count).reduce(0, +)) / secondsSeen
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
        for (index, name) in features.enumerated() {
            // Clamp to ±3 standard deviations: the model never saw values beyond that,
            // so it shouldn't extrapolate wildly from an unusual first few seconds.
            let standardized = min(max(((values[name] ?? means[index]) - means[index]) / scales[index], -3), 3)
            contributions[name] = weights[index] * standardized
            total += weights[index] * standardized
        }
        return (exp(total), contributions)
    }

    /// Plain-English reasons behind the prediction, biggest first.
    func explain(_ values: [String: Double], _ contributions: [String: Double]) -> [Explanation] {
        let percent = { (name: String) in Int(((values[name] ?? 0) * 100).rounded()) }
        let phrases: [String: (up: String, down: String)] = [
            "cottonoid_amt": ("Cottonoid in use \(percent("cottonoid_amt"))% of the time",
                               "Cottonoid in use \(percent("cottonoid_amt"))% of the time"),
            "blood_late": ("Field \(percent("blood_late"))% obscured by blood",
                           "Field clearing, \(percent("blood_late"))% blood"),
            "blood_mean": ("Blood averaging \(percent("blood_mean"))% of the field",
                           "Blood averaging \(percent("blood_mean"))% of the field"),
            "tools_per_frame": ("Few instruments in the field", "Multiple instruments in use"),
            "suction_amt": ("Suction visible \(percent("suction_amt"))% of the time",
                             "Suction visible \(percent("suction_amt"))% of the time"),
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

    private func slope(x times: [Double], y values: [Double]) -> Double {
        // Least-squares line through (time, blood): its slope says how fast the bleeding is changing
        guard times.count > 1 else { return 0 }
        let averageTime = times.reduce(0, +) / Double(times.count)
        let averageValue = values.reduce(0, +) / Double(values.count)
        let covariance = zip(times, values).map { ($0 - averageTime) * ($1 - averageValue) }.reduce(0, +)
        let timeSpread = times.map { ($0 - averageTime) * ($0 - averageTime) }.reduce(0, +)
        return timeSpread == 0 ? 0 : covariance / timeSpread
    }
}
