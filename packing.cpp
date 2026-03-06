#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <csignal>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kNumSemicircles = 15;
constexpr int kParamDim = 3 * kNumSemicircles;
constexpr int kFivefoldDim = 8;
constexpr int kOrbitBreakHarmonics = 4;
constexpr int kOrbitBreakProperties = 3;
constexpr int kOrbitBreakOrbits = 2;
constexpr int kOrbitBreakPerOrbit = kOrbitBreakHarmonics * kOrbitBreakProperties;
constexpr int kOrbitBreakDim = kOrbitBreakOrbits * kOrbitBreakPerOrbit;
constexpr int kMaxRingCount = 3;
constexpr int kStrategyCount = 6;
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kTwoPi = 2.0 * kPi;
constexpr double kValidMarginTol = -1e-9;
constexpr double kContainTol = 1e-9;
constexpr int kInitialBoundarySamples = 64;
constexpr int kValidateArcPoints = 256;
constexpr double kOverlapTol = 1e-6;

struct Vec2 {
    double x = 0.0;
    double y = 0.0;
};

struct Semicircle {
    double x = 0.0;
    double y = 0.0;
    double theta = 0.0;
};

struct Circle {
    double x = 0.0;
    double y = 0.0;
    double r = 0.0;
};

struct RadiusEval {
    double radius = 0.0;
    double grad_x = 0.0;
    double grad_y = 0.0;
    double grad_theta = 0.0;
};

struct SepInfo {
    double margin = 0.0;
    double phi = 0.0;
    int mode_a = 0;
    int mode_b = 0;
};

struct ObjectivePhase {
    double lambda = 0.0;
    double beta = 0.0;
    double tau = 0.0;
    double margin_target = 0.0;
    int max_iters = 0;
    int recenter_period = 1;
};

struct ObjectiveSummary {
    double cost = 0.0;
    double exact_radius = 0.0;
    double min_margin = std::numeric_limits<double>::infinity();
};

struct ObjectiveEval {
    double cost = 0.0;
    double exact_radius = 0.0;
    double min_margin = std::numeric_limits<double>::infinity();
    std::array<double, kParamDim> grad{};
};

struct OrbitBreakEval {
    double cost = 0.0;
    double exact_radius = 0.0;
    double min_margin = std::numeric_limits<double>::infinity();
    std::array<double, kOrbitBreakDim> grad{};
};

struct ValidationSummary {
    bool valid = false;
    double score = std::numeric_limits<double>::infinity();
    double min_margin = std::numeric_limits<double>::infinity();
    Circle mec{};
};

enum class StrategyId {
    StrictArchiveRefine = 0,
    FivefoldExact = 1,
    FivefoldBroken = 2,
    RingPartition = 3,
    CompressedContinuation = 4,
    RelaxedWave = 5,
};

struct StrategyCounters {
    std::uint64_t attempts = 0;
    std::uint64_t official_improvements = 0;
    std::uint64_t relaxed_inserts = 0;
};

struct CandidateRecord {
    std::array<Semicircle, kNumSemicircles> layout{};
    double score = std::numeric_limits<double>::infinity();
    double min_margin = std::numeric_limits<double>::infinity();
    std::string strategy;
    std::string candidate_kind;
    std::string key;
};

struct RingFamilySpec {
    const char* name = "";
    std::array<int, kMaxRingCount> counts{{0, 0, 0}};
    int ring_count = 0;
};

enum class RingOrientationPreset {
    Inward = 0,
    TangentCW = 1,
    Alternating = 2,
};

struct HomotopyPhase {
    double radius = 1.0;
    double overlap_weight = 0.0;
    double margin_weight = 0.0;
    double beta = 0.0;
    double margin_target = 0.0;
    double center_weight = 0.0;
    int iterations = 0;
    double step_xy = 0.0;
    double step_theta = 0.0;
    double scale_step = 0.0;
    double start_temp = 0.0;
    double end_temp = 0.0;
    int arc_points = 0;
};

struct HomotopyEval {
    double cost = std::numeric_limits<double>::infinity();
    double mec_radius = std::numeric_limits<double>::infinity();
    double overlap = 0.0;
    double margin_penalty = 0.0;
    double min_margin = std::numeric_limits<double>::infinity();
    Circle mec{};
};

using Layout = std::array<Semicircle, kNumSemicircles>;
using ParamVector = std::array<double, kParamDim>;
using FivefoldVector = std::array<double, kFivefoldDim>;
using OrbitBreakVector = std::array<double, kOrbitBreakDim>;

struct SharedBest {
    std::mutex mutex;
    Layout layout{};
    double score = std::numeric_limits<double>::infinity();
    bool has_valid = false;
    std::chrono::steady_clock::time_point start_time;
    std::string history_path;
    std::string official_archive_path;
    std::string relaxed_archive_path;
    std::size_t archive_size = 24;
    std::size_t relaxed_archive_size = 64;
    std::vector<CandidateRecord> official_archive;
    std::vector<CandidateRecord> relaxed_archive;
    std::array<StrategyCounters, kStrategyCount> strategy_counters{};
};

struct Options {
    int threads = 0;
    double seconds = 60.0;
    bool run_until_interrupt = false;
    int checkpoint_seconds = 300;
    bool no_resume = false;
    std::size_t archive_size = 24;
    std::size_t relaxed_archive_size = 64;
    std::string input_path;
    std::string output_path = "solution.json";
    std::string history_path;
};

struct BestSnapshot {
    bool has_valid = false;
    Layout layout{};
    double score = std::numeric_limits<double>::infinity();
    std::vector<CandidateRecord> official_archive;
    std::vector<CandidateRecord> relaxed_archive;
    std::array<StrategyCounters, kStrategyCount> strategy_counters{};
};

Layout rotate_layout(Layout layout, double angle);
Layout scale_about_point(Layout layout, const Vec2& center, double scale);
ObjectiveSummary objective_value(const ParamVector& x, const ObjectivePhase& phase);

std::atomic<bool> g_stop_requested{false};

void handle_stop_signal(int) {
    g_stop_requested.store(true, std::memory_order_relaxed);
}

bool search_should_stop(std::chrono::steady_clock::time_point deadline) {
    return g_stop_requested.load(std::memory_order_relaxed) ||
           std::chrono::steady_clock::now() >= deadline;
}

double sqr(double x) {
    return x * x;
}

double norm(const Vec2& v) {
    return std::hypot(v.x, v.y);
}

double distance(const Vec2& a, const Vec2& b) {
    return std::hypot(a.x - b.x, a.y - b.y);
}

double normalize_angle(double theta) {
    theta = std::fmod(theta, kTwoPi);
    if (theta <= -kPi) {
        theta += kTwoPi;
    } else if (theta > kPi) {
        theta -= kTwoPi;
    }
    return theta;
}

double mod_two_pi(double theta) {
    theta = std::fmod(theta, kTwoPi);
    if (theta < 0.0) {
        theta += kTwoPi;
    }
    return theta;
}

double round6(double value) {
    return std::round(value * 1'000'000.0) / 1'000'000.0;
}

Semicircle rounded_shape(const Semicircle& sc) {
    return {
        round6(sc.x),
        round6(sc.y),
        round6(normalize_angle(sc.theta)),
    };
}

Layout round_layout(Layout layout) {
    for (Semicircle& sc : layout) {
        sc = rounded_shape(sc);
    }
    return layout;
}

ParamVector to_vector(const Layout& layout) {
    ParamVector x{};
    for (int i = 0; i < kNumSemicircles; ++i) {
        x[3 * i + 0] = layout[i].x;
        x[3 * i + 1] = layout[i].y;
        x[3 * i + 2] = layout[i].theta;
    }
    return x;
}

Layout to_layout(const ParamVector& x) {
    Layout layout{};
    for (int i = 0; i < kNumSemicircles; ++i) {
        layout[i] = {
            x[3 * i + 0],
            x[3 * i + 1],
            normalize_angle(x[3 * i + 2]),
        };
    }
    return layout;
}

void normalize_angles(ParamVector& x) {
    for (int i = 0; i < kNumSemicircles; ++i) {
        x[3 * i + 2] = normalize_angle(x[3 * i + 2]);
    }
}

double dot_product(const ParamVector& a, const ParamVector& b) {
    double sum = 0.0;
    for (int i = 0; i < kParamDim; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

double l2_norm(const ParamVector& x) {
    return std::sqrt(dot_product(x, x));
}

void axpy(ParamVector& y, double alpha, const ParamVector& x) {
    for (int i = 0; i < kParamDim; ++i) {
        y[i] += alpha * x[i];
    }
}

ParamVector scaled(const ParamVector& x, double alpha) {
    ParamVector out{};
    for (int i = 0; i < kParamDim; ++i) {
        out[i] = alpha * x[i];
    }
    return out;
}

ParamVector subtract(const ParamVector& a, const ParamVector& b) {
    ParamVector out{};
    for (int i = 0; i < kParamDim; ++i) {
        out[i] = a[i] - b[i];
    }
    return out;
}

ParamVector add_scaled(const ParamVector& x, const ParamVector& p, double step) {
    ParamVector out{};
    for (int i = 0; i < kParamDim; ++i) {
        out[i] = x[i] + step * p[i];
    }
    return out;
}

double dot_product(const FivefoldVector& a, const FivefoldVector& b) {
    double sum = 0.0;
    for (int i = 0; i < kFivefoldDim; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

double l2_norm(const FivefoldVector& x) {
    return std::sqrt(dot_product(x, x));
}

void axpy(FivefoldVector& y, double alpha, const FivefoldVector& x) {
    for (int i = 0; i < kFivefoldDim; ++i) {
        y[i] += alpha * x[i];
    }
}

FivefoldVector scaled(const FivefoldVector& x, double alpha) {
    FivefoldVector out{};
    for (int i = 0; i < kFivefoldDim; ++i) {
        out[i] = alpha * x[i];
    }
    return out;
}

FivefoldVector subtract(const FivefoldVector& a, const FivefoldVector& b) {
    FivefoldVector out{};
    for (int i = 0; i < kFivefoldDim; ++i) {
        out[i] = a[i] - b[i];
    }
    return out;
}

FivefoldVector add_scaled(const FivefoldVector& x, const FivefoldVector& p, double step) {
    FivefoldVector out{};
    for (int i = 0; i < kFivefoldDim; ++i) {
        out[i] = x[i] + step * p[i];
    }
    return out;
}

double dot_product(const OrbitBreakVector& a, const OrbitBreakVector& b) {
    double sum = 0.0;
    for (int i = 0; i < kOrbitBreakDim; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

double l2_norm(const OrbitBreakVector& x) {
    return std::sqrt(dot_product(x, x));
}

void axpy(OrbitBreakVector& y, double alpha, const OrbitBreakVector& x) {
    for (int i = 0; i < kOrbitBreakDim; ++i) {
        y[i] += alpha * x[i];
    }
}

OrbitBreakVector scaled(const OrbitBreakVector& x, double alpha) {
    OrbitBreakVector out{};
    for (int i = 0; i < kOrbitBreakDim; ++i) {
        out[i] = alpha * x[i];
    }
    return out;
}

OrbitBreakVector subtract(const OrbitBreakVector& a, const OrbitBreakVector& b) {
    OrbitBreakVector out{};
    for (int i = 0; i < kOrbitBreakDim; ++i) {
        out[i] = a[i] - b[i];
    }
    return out;
}

OrbitBreakVector add_scaled(const OrbitBreakVector& x, const OrbitBreakVector& p, double step) {
    OrbitBreakVector out{};
    for (int i = 0; i < kOrbitBreakDim; ++i) {
        out[i] = x[i] + step * p[i];
    }
    return out;
}

const char* strategy_name(StrategyId strategy) {
    switch (strategy) {
        case StrategyId::StrictArchiveRefine: return "strict_archive_refine";
        case StrategyId::FivefoldExact: return "fivefold_exact";
        case StrategyId::FivefoldBroken: return "fivefold_broken";
        case StrategyId::RingPartition: return "ring_partition";
        case StrategyId::CompressedContinuation: return "compressed_continuation";
        case StrategyId::RelaxedWave: return "relaxed_wave";
    }
    return "unknown";
}

const char* ring_orientation_name(RingOrientationPreset preset) {
    switch (preset) {
        case RingOrientationPreset::Inward: return "inward";
        case RingOrientationPreset::TangentCW: return "tangent_cw";
        case RingOrientationPreset::Alternating: return "alternating";
    }
    return "unknown";
}

const std::array<RingFamilySpec, 5>& ring_family_specs() {
    static const std::array<RingFamilySpec, 5> specs{{
        {"5+5+5", {{5, 5, 5}}, 3},
        {"4+5+6", {{4, 5, 6}}, 3},
        {"3+6+6", {{3, 6, 6}}, 3},
        {"5+10", {{5, 10, 0}}, 2},
        {"6+9", {{6, 9, 0}}, 2},
    }};
    return specs;
}

std::string sidecar_path(const std::string& path, const std::string& suffix) {
    const std::size_t slash = path.find_last_of("/\\");
    const std::size_t dot = path.find_last_of('.');
    if (dot != std::string::npos && (slash == std::string::npos || dot > slash)) {
        return path.substr(0, dot) + suffix;
    }
    return path + suffix;
}

std::string layout_key(const Layout& layout) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    for (int i = 0; i < kNumSemicircles; ++i) {
        const Semicircle sc = rounded_shape(layout[i]);
        if (i != 0) {
            oss << ';';
        }
        oss << sc.x << ',' << sc.y << ',' << sc.theta;
    }
    return oss.str();
}

void log_history_event(
    SharedBest& shared,
    double score,
    int thread_id,
    const std::string& strategy,
    const std::string& candidate_kind,
    const std::string& archive_event
) {
    if (shared.history_path.empty()) {
        return;
    }

    std::ofstream history(shared.history_path, std::ios::app);
    if (!history) {
        throw std::runtime_error("Failed to open history file: " + shared.history_path);
    }

    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - shared.start_time).count();
    history << std::fixed << std::setprecision(6)
            << elapsed << '\t'
            << score << '\t'
            << thread_id << '\t'
            << strategy << '\t'
            << candidate_kind << '\t'
            << archive_event << '\n';
}

bool insert_candidate_archive(
    std::vector<CandidateRecord>& archive,
    CandidateRecord candidate,
    std::size_t limit
) {
    candidate.key = layout_key(candidate.layout);
    auto it = std::find_if(archive.begin(), archive.end(), [&](const CandidateRecord& entry) {
        return entry.key == candidate.key;
    });

    if (it != archive.end()) {
        if (candidate.score + 1e-12 >= it->score) {
            return false;
        }
        *it = std::move(candidate);
    } else {
        archive.push_back(std::move(candidate));
    }

    std::sort(archive.begin(), archive.end(), [](const CandidateRecord& lhs, const CandidateRecord& rhs) {
        if (std::abs(lhs.score - rhs.score) > 1e-12) {
            return lhs.score < rhs.score;
        }
        return lhs.key < rhs.key;
    });
    if (archive.size() > limit) {
        archive.resize(limit);
    }
    return true;
}

double dot_product(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

double l2_norm(const std::vector<double>& x) {
    return std::sqrt(dot_product(x, x));
}

void axpy(std::vector<double>& y, double alpha, const std::vector<double>& x) {
    for (std::size_t i = 0; i < y.size(); ++i) {
        y[i] += alpha * x[i];
    }
}

std::vector<double> scaled(const std::vector<double>& x, double alpha) {
    std::vector<double> out = x;
    for (double& value : out) {
        value *= alpha;
    }
    return out;
}

std::vector<double> subtract(const std::vector<double>& a, const std::vector<double>& b) {
    std::vector<double> out(a.size(), 0.0);
    for (std::size_t i = 0; i < a.size(); ++i) {
        out[i] = a[i] - b[i];
    }
    return out;
}

std::vector<double> add_scaled(const std::vector<double>& x, const std::vector<double>& p, double step) {
    std::vector<double> out(x.size(), 0.0);
    for (std::size_t i = 0; i < x.size(); ++i) {
        out[i] = x[i] + step * p[i];
    }
    return out;
}

template <typename Builder>
ObjectiveSummary reduced_objective_value(
    const std::vector<double>& p,
    const ObjectivePhase& phase,
    const Builder& builder
) {
    return objective_value(to_vector(builder(p)), phase);
}

template <typename Builder, typename Normalizer>
std::vector<double> reduced_gradient(
    std::vector<double> p,
    const ObjectivePhase& phase,
    const Builder& builder,
    const Normalizer& normalizer
) {
    std::vector<double> grad(p.size(), 0.0);
    const double f0 = reduced_objective_value(p, phase, builder).cost;
    for (std::size_t i = 0; i < p.size(); ++i) {
        const double h = 1e-5 * (1.0 + std::abs(p[i]));
        std::vector<double> plus = p;
        std::vector<double> minus = p;
        plus[i] += h;
        minus[i] -= h;
        normalizer(plus);
        normalizer(minus);
        const double fp = reduced_objective_value(plus, phase, builder).cost;
        const double fm = reduced_objective_value(minus, phase, builder).cost;
        grad[i] = (fp - fm) / (2.0 * h);
    }
    return grad;
}

std::vector<double> lbfgs_direction(
    const std::vector<double>& grad,
    const std::vector<std::vector<double>>& S,
    const std::vector<std::vector<double>>& Y,
    const std::vector<double>& rho
) {
    const int m = static_cast<int>(S.size());
    std::vector<double> q = grad;
    std::vector<double> alpha(static_cast<std::size_t>(m), 0.0);

    for (int i = m - 1; i >= 0; --i) {
        alpha[static_cast<std::size_t>(i)] =
            rho[static_cast<std::size_t>(i)] * dot_product(S[static_cast<std::size_t>(i)], q);
        axpy(q, -alpha[static_cast<std::size_t>(i)], Y[static_cast<std::size_t>(i)]);
    }

    double gamma = 1.0;
    if (m > 0) {
        const double sy = dot_product(S.back(), Y.back());
        const double yy = dot_product(Y.back(), Y.back());
        if (yy > 1e-20) {
            gamma = sy / yy;
        }
    }

    std::vector<double> r = scaled(q, gamma);
    for (int i = 0; i < m; ++i) {
        const double beta = rho[static_cast<std::size_t>(i)] * dot_product(Y[static_cast<std::size_t>(i)], r);
        axpy(r, alpha[static_cast<std::size_t>(i)] - beta, S[static_cast<std::size_t>(i)]);
    }
    for (double& value : r) {
        value = -value;
    }
    return r;
}

template <typename Builder, typename Normalizer>
std::vector<double> optimize_reduced(
    std::vector<double> p,
    const std::vector<ObjectivePhase>& phases,
    std::chrono::steady_clock::time_point deadline,
    const Builder& builder,
    const Normalizer& normalizer
) {
    normalizer(p);
    constexpr std::size_t kMemoryLimit = 8;

    for (const ObjectivePhase& phase : phases) {
        std::vector<std::vector<double>> S;
        std::vector<std::vector<double>> Y;
        std::vector<double> rho;
        ObjectiveSummary value = reduced_objective_value(p, phase, builder);
        std::vector<double> grad = reduced_gradient(p, phase, builder, normalizer);

        const int max_iters = std::max(12, phase.max_iters / 2);
        for (int iter = 0; iter < max_iters; ++iter) {
            if (search_should_stop(deadline)) {
                return p;
            }

            std::vector<double> direction = lbfgs_direction(grad, S, Y, rho);
            double gtp = dot_product(grad, direction);
            if (!(gtp < -1e-12) || !std::isfinite(gtp)) {
                direction = scaled(grad, -1.0);
                gtp = -dot_product(grad, grad);
            }

            const double dir_norm = l2_norm(direction);
            if (!std::isfinite(dir_norm) || dir_norm < 1e-14) {
                break;
            }

            if (dir_norm > 0.25) {
                direction = scaled(direction, 0.25 / dir_norm);
                gtp = dot_product(grad, direction);
            }

            double step = 1.0;
            bool accepted = false;
            std::vector<double> p_new;
            ObjectiveSummary value_new{};
            for (int line_search = 0; line_search < 20; ++line_search) {
                p_new = add_scaled(p, direction, step);
                normalizer(p_new);
                value_new = reduced_objective_value(p_new, phase, builder);
                if (std::isfinite(value_new.cost) &&
                    value_new.cost <= value.cost + 1e-4 * step * gtp) {
                    accepted = true;
                    break;
                }
                step *= 0.5;
            }
            if (!accepted) {
                break;
            }

            std::vector<double> grad_new = reduced_gradient(p_new, phase, builder, normalizer);
            const std::vector<double> s = subtract(p_new, p);
            const std::vector<double> y = subtract(grad_new, grad);
            const double sy = dot_product(s, y);
            if (sy > 1e-12 && std::isfinite(sy)) {
                if (S.size() == kMemoryLimit) {
                    S.erase(S.begin());
                    Y.erase(Y.begin());
                    rho.erase(rho.begin());
                }
                S.push_back(s);
                Y.push_back(y);
                rho.push_back(1.0 / sy);
            }

            p = std::move(p_new);
            value = value_new;
            grad = std::move(grad_new);
            if (l2_norm(grad) < 1e-7) {
                break;
            }
        }
    }

    return p;
}

void note_strategy_attempt(SharedBest& shared, StrategyId strategy) {
    std::scoped_lock lock(shared.mutex);
    shared.strategy_counters[static_cast<int>(strategy)].attempts += 1;
}

bool maybe_publish_valid(
    SharedBest& shared,
    const Layout& layout,
    const ValidationSummary& validation,
    int thread_id,
    StrategyId strategy_id = StrategyId::StrictArchiveRefine,
    const std::string& strategy = "bootstrap"
) {
    const Layout rounded = round_layout(layout);

    std::scoped_lock lock(shared.mutex);
    CandidateRecord candidate;
    candidate.layout = rounded;
    candidate.score = validation.score;
    candidate.min_margin = validation.min_margin;
    candidate.strategy = strategy;
    candidate.candidate_kind = "official";

    const bool archive_inserted = insert_candidate_archive(
        shared.official_archive,
        candidate,
        shared.archive_size
    );
    if (archive_inserted) {
        log_history_event(shared, validation.score, thread_id, strategy, "official", "official_archive_insert");
    }

    if (!shared.has_valid || validation.score + 1e-12 < shared.score) {
        shared.layout = rounded;
        shared.score = validation.score;
        shared.has_valid = true;
        shared.strategy_counters[static_cast<int>(strategy_id)].official_improvements += 1;
        log_history_event(shared, validation.score, thread_id, strategy, "official", "incumbent_update");

        const double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - shared.start_time).count();
        std::cout << "[t" << thread_id << "] best valid score "
                  << std::fixed << std::setprecision(6) << validation.score
                  << "  elapsed=" << std::setprecision(2) << elapsed << "s"
                  << "  strategy=" << strategy
                  << std::endl;
    }

    return archive_inserted;
}

bool maybe_record_relaxed_candidate(
    SharedBest& shared,
    const Layout& layout,
    double relaxed_score,
    double min_margin,
    int thread_id,
    StrategyId strategy_id,
    const std::string& strategy
) {
    const Layout rounded = round_layout(layout);

    std::scoped_lock lock(shared.mutex);
    CandidateRecord candidate;
    candidate.layout = rounded;
    candidate.score = relaxed_score;
    candidate.min_margin = min_margin;
    candidate.strategy = strategy;
    candidate.candidate_kind = "relaxed";

    const bool archive_inserted = insert_candidate_archive(
        shared.relaxed_archive,
        candidate,
        shared.relaxed_archive_size
    );
    if (archive_inserted) {
        shared.strategy_counters[static_cast<int>(strategy_id)].relaxed_inserts += 1;
        log_history_event(shared, relaxed_score, thread_id, strategy, "relaxed", "relaxed_archive_insert");
    }
    return archive_inserted;
}

void initialize_history_log(const std::string& path, bool reset_file) {
    if (path.empty()) {
        return;
    }
    if (!reset_file) {
        std::ifstream existing(path);
        if (existing.good()) {
            return;
        }
    }
    std::ofstream history(path);
    if (!history) {
        throw std::runtime_error("Failed to open history file: " + path);
    }
    history << "elapsed_seconds\tscore\tthread_id\tstrategy\tcandidate_kind\tarchive_event\n";
}

BestSnapshot snapshot_best_state(SharedBest& shared) {
    std::scoped_lock lock(shared.mutex);
    return {
        shared.has_valid,
        shared.layout,
        shared.score,
        shared.official_archive,
        shared.relaxed_archive,
        shared.strategy_counters,
    };
}

std::optional<Layout> snapshot_best(SharedBest& shared) {
    std::scoped_lock lock(shared.mutex);
    if (!shared.has_valid) {
        return std::nullopt;
    }
    return shared.layout;
}

std::vector<CandidateRecord> snapshot_official_archive(SharedBest& shared) {
    std::scoped_lock lock(shared.mutex);
    return shared.official_archive;
}

std::vector<CandidateRecord> snapshot_relaxed_archive(SharedBest& shared) {
    std::scoped_lock lock(shared.mutex);
    return shared.relaxed_archive;
}

Layout known_best_solution() {
    return {{
        {0.914308, 0.000000, 1.317821},
        {1.966568, 0.157031, 0.496516},
        {1.625939, 1.117275, 1.718923},
        {0.282537, 0.869559, 2.574458},
        {0.458358, 1.918842, 1.753153},
        {-0.560149, 1.891617, 2.975561},
        {-0.739691, 0.537417, -2.452090},
        {-1.683287, 1.028879, 3.009790},
        {-1.972130, 0.051808, -2.050988},
        {-0.739691, -0.537417, -1.195453},
        {-1.498686, -1.282960, -2.016758},
        {-0.658694, -1.859598, -0.794351},
        {0.282537, -0.869559, 0.061184},
        {0.757048, -1.821792, -0.760121},
        {1.565035, -1.201103, 0.462286},
    }};
}

Layout legacy_solution() {
    return {{
        {0.885253, 0.048048, 1.258115},
        {1.747063, 1.164639, 1.664576},
        {2.098517, 0.028988, 0.543258},
        {0.237928, 0.914999, 2.469045},
        {-0.583090, 2.000441, 2.810600},
        {0.579869, 2.018007, 1.790468},
        {-0.729711, 0.495978, -2.511797},
        {-2.047033, 0.131172, -2.169404},
        {-1.743154, 1.147344, 3.111508},
        {-0.706395, -0.667941, -1.372357},
        {-0.688939, -1.947947, -0.963381},
        {-1.626354, -1.328000, -2.034400},
        {0.377529, -0.886796, -0.007017},
        {1.662523, -1.247409, 0.337349},
        {0.761111, -1.942384, -0.782925},
    }};
}

Layout fivefold_seed() {
    constexpr double r1 = 0.9143084400146972;
    constexpr double t1 = 1.3178213000150536;
    constexpr double r2 = 1.9728274236077763;
    constexpr double d2 = 0.0796811625821883;
    constexpr double t2 = 0.4965162226010280;
    constexpr double r3 = 1.9728106612272804;
    constexpr double d3 = 0.60205433;
    constexpr double t3 = 1.7189234808277160;

    Layout layout{};
    int idx = 0;
    for (int k = 0; k < 5; ++k) {
        const double base = kTwoPi * static_cast<double>(k) / 5.0;
        layout[idx++] = {r1 * std::cos(base), r1 * std::sin(base), normalize_angle(base + t1)};
        layout[idx++] = {r2 * std::cos(base + d2), r2 * std::sin(base + d2), normalize_angle(base + t2)};
        layout[idx++] = {r3 * std::cos(base + d3), r3 * std::sin(base + d3), normalize_angle(base + t3)};
    }
    return layout;
}

FivefoldVector fivefold_params() {
    return {{
        0.9143084400146972,
        1.3178213000150536,
        1.9728274236077763,
        0.0796811625821883,
        0.4965162226010280,
        1.9728106612272804,
        0.6020543300146995,
        1.7189234808277160,
    }};
}

std::array<double, kOrbitBreakHarmonics> orbit_break_basis(int k) {
    const double angle = kTwoPi * static_cast<double>(k) / 5.0;
    return {{
        std::cos(angle),
        std::sin(angle),
        std::cos(2.0 * angle),
        std::sin(2.0 * angle),
    }};
}

double orbit_break_combo(
    const OrbitBreakVector& q,
    int offset,
    const std::array<double, kOrbitBreakHarmonics>& basis
) {
    double value = 0.0;
    for (int i = 0; i < kOrbitBreakHarmonics; ++i) {
        value += q[offset + i] * basis[i];
    }
    return value;
}

void normalize_orbit_break(OrbitBreakVector& q) {
    for (int orbit = 0; orbit < kOrbitBreakOrbits; ++orbit) {
        const int base = orbit * kOrbitBreakPerOrbit;
        for (int i = 0; i < kOrbitBreakHarmonics; ++i) {
            q[base + i] = std::clamp(q[base + i], -0.040, 0.040);
            q[base + 4 + i] = std::clamp(q[base + 4 + i], -0.050, 0.050);
            q[base + 8 + i] = std::clamp(q[base + 8 + i], -0.080, 0.080);
        }
    }
}

Layout orbit_break_layout_from_params(const FivefoldVector& p, const OrbitBreakVector& q) {
    Layout layout{};
    int idx = 0;
    for (int k = 0; k < 5; ++k) {
        const double base = kTwoPi * static_cast<double>(k) / 5.0;
        const auto basis = orbit_break_basis(k);

        layout[idx++] = {
            p[0] * std::cos(base),
            p[0] * std::sin(base),
            normalize_angle(base + p[1]),
        };

        for (int orbit = 0; orbit < kOrbitBreakOrbits; ++orbit) {
            const int q_base = orbit * kOrbitBreakPerOrbit;
            const double orbit_r = orbit == 0 ? p[2] : p[5];
            const double orbit_angle = orbit == 0 ? p[3] : p[6];
            const double orbit_theta = orbit == 0 ? p[4] : p[7];
            const double dr = orbit_break_combo(q, q_base + 0, basis);
            const double da = orbit_break_combo(q, q_base + 4, basis);
            const double dt = orbit_break_combo(q, q_base + 8, basis);
            const double radius = orbit_r + dr;
            const double angle = base + orbit_angle + da;

            layout[idx++] = {
                radius * std::cos(angle),
                radius * std::sin(angle),
                normalize_angle(base + orbit_theta + dt),
            };
        }
    }
    return layout;
}

Layout fivefold_layout_from_params(const FivefoldVector& p) {
    Layout layout{};
    int idx = 0;
    for (int k = 0; k < 5; ++k) {
        const double base = kTwoPi * static_cast<double>(k) / 5.0;
        layout[idx++] = {
            p[0] * std::cos(base),
            p[0] * std::sin(base),
            normalize_angle(base + p[1]),
        };
        layout[idx++] = {
            p[2] * std::cos(base + p[3]),
            p[2] * std::sin(base + p[3]),
            normalize_angle(base + p[4]),
        };
        layout[idx++] = {
            p[5] * std::cos(base + p[6]),
            p[5] * std::sin(base + p[6]),
            normalize_angle(base + p[7]),
        };
    }
    return layout;
}

int ring_family_param_dim(const RingFamilySpec& spec) {
    int dim = 0;
    for (int ring = 0; ring < spec.ring_count; ++ring) {
        dim += 3;
        if (spec.counts[ring] >= 4) {
            dim += 6;
        }
    }
    return dim;
}

std::vector<double> ring_family_initial_params(
    const RingFamilySpec& spec,
    RingOrientationPreset preset,
    std::mt19937_64& rng
) {
    std::vector<double> params(static_cast<std::size_t>(ring_family_param_dim(spec)), 0.0);
    std::normal_distribution<double> jitter_r(0.0, 0.06);
    std::normal_distribution<double> jitter_a(0.0, 0.08);
    std::size_t idx = 0;
    for (int ring = 0; ring < spec.ring_count; ++ring) {
        params[idx++] = 0.90 + 0.55 * static_cast<double>(ring) + jitter_r(rng);
        params[idx++] = 0.18 * static_cast<double>(ring) + jitter_a(rng);
        (void)preset;
        params[idx++] = 0.05 * jitter_a(rng);
        if (spec.counts[ring] >= 4) {
            idx += 6;
        }
    }
    return params;
}

void normalize_ring_family_params(std::vector<double>& params, const RingFamilySpec& spec) {
    std::size_t idx = 0;
    for (int ring = 0; ring < spec.ring_count; ++ring) {
        params[idx] = std::clamp(params[idx], 0.25, 3.2);
        params[idx + 1] = normalize_angle(params[idx + 1]);
        params[idx + 2] = normalize_angle(params[idx + 2]);
        idx += 3;
        if (spec.counts[ring] >= 4) {
            for (int k = 0; k < 2; ++k) {
                params[idx + k] = std::clamp(params[idx + k], -0.18, 0.18);
            }
            for (int k = 2; k < 4; ++k) {
                params[idx + k] = std::clamp(params[idx + k], -0.22, 0.22);
            }
            for (int k = 4; k < 6; ++k) {
                params[idx + k] = std::clamp(params[idx + k], -0.35, 0.35);
            }
            idx += 6;
        }
    }
}

Layout ring_family_layout_from_params(
    const RingFamilySpec& spec,
    RingOrientationPreset preset,
    const std::vector<double>& params
) {
    Layout layout{};
    int shape_idx = 0;
    std::size_t idx = 0;

    for (int ring = 0; ring < spec.ring_count; ++ring) {
        const int count = spec.counts[ring];
        const double radius = params[idx++];
        const double phase = params[idx++];
        const double theta_offset = params[idx++];

        double dr_cos = 0.0;
        double dr_sin = 0.0;
        double da_cos = 0.0;
        double da_sin = 0.0;
        double dt_cos = 0.0;
        double dt_sin = 0.0;
        if (count >= 4) {
            dr_cos = params[idx++];
            dr_sin = params[idx++];
            da_cos = params[idx++];
            da_sin = params[idx++];
            dt_cos = params[idx++];
            dt_sin = params[idx++];
        }

        for (int j = 0; j < count; ++j) {
            const double local = kTwoPi * static_cast<double>(j) / static_cast<double>(count);
            const double wave_cos = std::cos(local);
            const double wave_sin = std::sin(local);
            const double radial = radius + dr_cos * wave_cos + dr_sin * wave_sin;
            const double angle = local + phase + da_cos * wave_cos + da_sin * wave_sin;
            double theta = angle + theta_offset + dt_cos * wave_cos + dt_sin * wave_sin;

            if (preset == RingOrientationPreset::Inward) {
                theta = angle + kPi + theta_offset + dt_cos * wave_cos + dt_sin * wave_sin;
            } else if (preset == RingOrientationPreset::TangentCW) {
                theta = angle - kPi * 0.5 + theta_offset + dt_cos * wave_cos + dt_sin * wave_sin;
            } else {
                const double base = (j % 2 == 0) ? angle - kPi * 0.5 : angle + kPi;
                theta = base + theta_offset + dt_cos * wave_cos + dt_sin * wave_sin;
            }

            layout[shape_idx++] = {
                radial * std::cos(angle),
                radial * std::sin(angle),
                normalize_angle(theta),
            };
        }
    }

    while (shape_idx < kNumSemicircles) {
        const double angle = kTwoPi * static_cast<double>(shape_idx) / static_cast<double>(kNumSemicircles);
        layout[shape_idx++] = {0.8 * std::cos(angle), 0.8 * std::sin(angle), normalize_angle(angle + kPi)};
    }
    return layout;
}

Layout random_shell_seed(std::mt19937_64& rng) {
    std::uniform_real_distribution<double> ur(0.0, 1.0);
    std::uniform_real_distribution<double> ua(-kPi, kPi);

    Layout layout{};
    for (int i = 0; i < kNumSemicircles; ++i) {
        const double r = 0.4 + 1.9 * std::sqrt(ur(rng));
        const double a = kTwoPi * ur(rng);
        layout[i] = {
            r * std::cos(a),
            r * std::sin(a),
            ua(rng),
        };
    }
    return layout;
}

std::optional<Layout> load_solution_json_from_contents(const std::string& contents) {
    std::vector<double> values;
    const char* ptr = contents.c_str();
    char* end = nullptr;
    while (*ptr != '\0') {
        if (std::isdigit(static_cast<unsigned char>(*ptr)) || *ptr == '-' || *ptr == '+' || *ptr == '.') {
            const double value = std::strtod(ptr, &end);
            if (end != ptr) {
                values.push_back(value);
                ptr = end;
                continue;
            }
        }
        ++ptr;
    }

    if (values.size() != static_cast<std::size_t>(kParamDim)) {
        return std::nullopt;
    }

    Layout layout{};
    for (int i = 0; i < kNumSemicircles; ++i) {
        layout[i] = rounded_shape({
            values[3 * i + 0],
            values[3 * i + 1],
            values[3 * i + 2],
        });
    }
    return layout;
}

std::optional<Layout> load_solution_json(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        return std::nullopt;
    }

    const std::string contents(
        (std::istreambuf_iterator<char>(in)),
        std::istreambuf_iterator<char>()
    );
    return load_solution_json_from_contents(contents);
}

void write_solution_json(const Layout& layout, const std::string& path) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to open output file: " + path);
    }

    out << std::fixed << std::setprecision(6);
    out << "[\n";
    for (int i = 0; i < kNumSemicircles; ++i) {
        const Semicircle sc = rounded_shape(layout[i]);
        out << "  { \"x\": " << sc.x
            << ", \"y\": " << sc.y
            << ", \"theta\": " << sc.theta << " }";
        out << (i + 1 == kNumSemicircles ? "\n" : ",\n");
    }
    out << "]\n";
}

void write_candidate_archive_json(const std::vector<CandidateRecord>& archive, const std::string& path) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to open archive file: " + path);
    }

    out << std::fixed << std::setprecision(6);
    out << "[\n";
    for (std::size_t i = 0; i < archive.size(); ++i) {
        const CandidateRecord& entry = archive[i];
        out << "  {\n";
        out << "    \"score\": " << entry.score << ",\n";
        out << "    \"min_margin\": " << entry.min_margin << ",\n";
        out << "    \"strategy\": \"" << entry.strategy << "\",\n";
        out << "    \"candidate_kind\": \"" << entry.candidate_kind << "\",\n";
        out << "    \"layout\": [\n";
        for (int j = 0; j < kNumSemicircles; ++j) {
            const Semicircle sc = rounded_shape(entry.layout[j]);
            out << "      { \"x\": " << sc.x
                << ", \"y\": " << sc.y
                << ", \"theta\": " << sc.theta << " }";
            out << (j + 1 == kNumSemicircles ? '\n' : ',');
            if (j + 1 != kNumSemicircles) {
                out << "\n";
            }
        }
        out << "    ]\n";
        out << "  }";
        out << (i + 1 == archive.size() ? "\n" : ",\n");
    }
    out << "]\n";
}

std::size_t find_matching_bracket(const std::string& text, std::size_t open_pos) {
    int depth = 0;
    for (std::size_t i = open_pos; i < text.size(); ++i) {
        if (text[i] == '[') {
            depth += 1;
        } else if (text[i] == ']') {
            depth -= 1;
            if (depth == 0) {
                return i;
            }
        }
    }
    return std::string::npos;
}

std::optional<double> parse_number_after_label(
    const std::string& text,
    std::size_t from,
    const std::string& label
) {
    const std::size_t label_pos = text.find(label, from);
    if (label_pos == std::string::npos) {
        return std::nullopt;
    }
    const std::size_t colon = text.find(':', label_pos + label.size());
    if (colon == std::string::npos) {
        return std::nullopt;
    }
    const char* begin = text.c_str() + colon + 1;
    char* end = nullptr;
    const double value = std::strtod(begin, &end);
    if (end == begin) {
        return std::nullopt;
    }
    return value;
}

std::optional<std::string> parse_string_after_label(
    const std::string& text,
    std::size_t from,
    const std::string& label
) {
    const std::size_t label_pos = text.find(label, from);
    if (label_pos == std::string::npos) {
        return std::nullopt;
    }
    const std::size_t first_quote = text.find('"', label_pos + label.size());
    if (first_quote == std::string::npos) {
        return std::nullopt;
    }
    const std::size_t second_quote = text.find('"', first_quote + 1);
    if (second_quote == std::string::npos) {
        return std::nullopt;
    }
    return text.substr(first_quote + 1, second_quote - first_quote - 1);
}

std::optional<std::vector<CandidateRecord>> load_candidate_archive_json(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        return std::nullopt;
    }

    const std::string contents(
        (std::istreambuf_iterator<char>(in)),
        std::istreambuf_iterator<char>()
    );

    std::vector<CandidateRecord> archive;
    std::size_t pos = 0;
    while (true) {
        const std::size_t score_pos = contents.find("\"score\"", pos);
        if (score_pos == std::string::npos) {
            break;
        }

        const auto score = parse_number_after_label(contents, score_pos, "\"score\"");
        const auto min_margin = parse_number_after_label(contents, score_pos, "\"min_margin\"");
        const auto strategy = parse_string_after_label(contents, score_pos, "\"strategy\"");
        const auto candidate_kind = parse_string_after_label(contents, score_pos, "\"candidate_kind\"");
        const std::size_t layout_label = contents.find("\"layout\"", score_pos);
        if (!score.has_value() || !min_margin.has_value() || !strategy.has_value() ||
            !candidate_kind.has_value() || layout_label == std::string::npos) {
            return std::nullopt;
        }

        const std::size_t layout_start = contents.find('[', layout_label);
        if (layout_start == std::string::npos) {
            return std::nullopt;
        }
        const std::size_t layout_end = find_matching_bracket(contents, layout_start);
        if (layout_end == std::string::npos) {
            return std::nullopt;
        }

        auto layout = load_solution_json_from_contents(contents.substr(layout_start, layout_end - layout_start + 1));
        if (!layout.has_value()) {
            return std::nullopt;
        }

        CandidateRecord entry;
        entry.layout = *layout;
        entry.score = *score;
        entry.min_margin = *min_margin;
        entry.strategy = *strategy;
        entry.candidate_kind = *candidate_kind;
        entry.key = layout_key(entry.layout);
        archive.push_back(entry);
        pos = layout_end + 1;
    }

    return archive;
}

void write_status_report(
    const BestSnapshot& best,
    const SharedBest& shared,
    const std::string& output_path,
    bool interrupted,
    bool final_report
) {
    if (!best.has_valid) {
        return;
    }

    write_solution_json(best.layout, output_path);
    write_candidate_archive_json(best.official_archive, shared.official_archive_path);
    write_candidate_archive_json(best.relaxed_archive, shared.relaxed_archive_path);

    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - shared.start_time).count();
    const std::string status_path = sidecar_path(output_path, ".status.txt");
    std::ofstream status(status_path);
    if (!status) {
        throw std::runtime_error("Failed to open status file: " + status_path);
    }

    status << std::fixed << std::setprecision(6);
    status << "score=" << best.score << "\n";
    status << "elapsed_seconds=" << elapsed << "\n";
    status << "interrupted=" << (interrupted ? "true" : "false") << "\n";
    status << "final_report=" << (final_report ? "true" : "false") << "\n";
    status << "official_archive_size=" << best.official_archive.size() << "\n";
    status << "relaxed_archive_size=" << best.relaxed_archive.size() << "\n";
    for (int i = 0; i < kStrategyCount; ++i) {
        status << strategy_name(static_cast<StrategyId>(i))
               << ".attempts=" << best.strategy_counters[static_cast<std::size_t>(i)].attempts << "\n";
        status << strategy_name(static_cast<StrategyId>(i))
               << ".official_improvements="
               << best.strategy_counters[static_cast<std::size_t>(i)].official_improvements << "\n";
        status << strategy_name(static_cast<StrategyId>(i))
               << ".relaxed_inserts="
               << best.strategy_counters[static_cast<std::size_t>(i)].relaxed_inserts << "\n";
    }
}

Circle circle_from_1(const Vec2& p) {
    return {p.x, p.y, 0.0};
}

Circle circle_from_2(const Vec2& a, const Vec2& b) {
    return {(a.x + b.x) * 0.5, (a.y + b.y) * 0.5, distance(a, b) * 0.5};
}

Circle circle_from_3(const Vec2& a, const Vec2& b, const Vec2& c) {
    const double d = 2.0 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y));
    if (std::abs(d) < 1e-14) {
        Circle best = circle_from_2(a, b);
        const Circle ab = circle_from_2(a, c);
        const Circle bc = circle_from_2(b, c);
        if (ab.r > best.r) {
            best = ab;
        }
        if (bc.r > best.r) {
            best = bc;
        }
        return best;
    }
    const double ux =
        ((sqr(a.x) + sqr(a.y)) * (b.y - c.y) +
         (sqr(b.x) + sqr(b.y)) * (c.y - a.y) +
         (sqr(c.x) + sqr(c.y)) * (a.y - b.y)) / d;
    const double uy =
        ((sqr(a.x) + sqr(a.y)) * (c.x - b.x) +
         (sqr(b.x) + sqr(b.y)) * (a.x - c.x) +
         (sqr(c.x) + sqr(c.y)) * (b.x - a.x)) / d;
    return {ux, uy, distance(a, {ux, uy})};
}

bool in_circle(const Circle& circle, const Vec2& p, double eps = 1e-10) {
    return distance({circle.x, circle.y}, p) <= circle.r + eps;
}

Circle minimum_enclosing_circle(std::vector<Vec2> pts) {
    std::mt19937_64 rng(42);
    std::shuffle(pts.begin(), pts.end(), rng);

    Circle circle{0.0, 0.0, -1.0};
    for (std::size_t i = 0; i < pts.size(); ++i) {
        if (circle.r >= 0.0 && in_circle(circle, pts[i])) {
            continue;
        }
        circle = circle_from_1(pts[i]);
        for (std::size_t j = 0; j < i; ++j) {
            if (in_circle(circle, pts[j])) {
                continue;
            }
            circle = circle_from_2(pts[i], pts[j]);
            for (std::size_t k = 0; k < j; ++k) {
                if (!in_circle(circle, pts[k])) {
                    circle = circle_from_3(pts[i], pts[j], pts[k]);
                }
            }
        }
    }
    if (circle.r < 0.0) {
        return {0.0, 0.0, 0.0};
    }
    return circle;
}

bool angle_on_arc(double angle, double theta) {
    const double diff = std::atan2(std::sin(angle - theta), std::cos(angle - theta));
    return diff >= -kPi * 0.5 - 1e-12 && diff <= kPi * 0.5 + 1e-12;
}

double cross(const Vec2& a, const Vec2& b) {
    return a.x * b.y - a.y * b.x;
}

Vec2 operator-(const Vec2& a, const Vec2& b) {
    return {a.x - b.x, a.y - b.y};
}

Vec2 operator+(const Vec2& a, const Vec2& b) {
    return {a.x + b.x, a.y + b.y};
}

Vec2 operator*(const Vec2& a, double scale) {
    return {a.x * scale, a.y * scale};
}

Vec2 endpoint_a(const Semicircle& sc) {
    const double angle = sc.theta - kPi * 0.5;
    return {sc.x + std::cos(angle), sc.y + std::sin(angle)};
}

Vec2 endpoint_b(const Semicircle& sc) {
    const double angle = sc.theta + kPi * 0.5;
    return {sc.x + std::cos(angle), sc.y + std::sin(angle)};
}

Vec2 endpoint_a_at(const Semicircle& sc, double radius) {
    const double angle = sc.theta - kPi * 0.5;
    return {sc.x + radius * std::cos(angle), sc.y + radius * std::sin(angle)};
}

Vec2 endpoint_b_at(const Semicircle& sc, double radius) {
    const double angle = sc.theta + kPi * 0.5;
    return {sc.x + radius * std::cos(angle), sc.y + radius * std::sin(angle)};
}

std::vector<Vec2> boundary_points(const Semicircle& sc, int n = kInitialBoundarySamples) {
    const int n_arc = n / 2;
    const int n_flat = n - n_arc;

    std::vector<Vec2> pts;
    pts.reserve(static_cast<std::size_t>(n));

    for (int i = 0; i < n_arc; ++i) {
        const double t = n_arc <= 1 ? 0.0 : static_cast<double>(i) / static_cast<double>(n_arc - 1);
        const double angle = sc.theta - kPi * 0.5 + kPi * t;
        pts.push_back({sc.x + std::cos(angle), sc.y + std::sin(angle)});
    }

    const Vec2 a = endpoint_a(sc);
    const Vec2 b = endpoint_b(sc);
    for (int i = 0; i < n_flat; ++i) {
        const double t = n_flat <= 1 ? 0.0 : static_cast<double>(i) / static_cast<double>(n_flat - 1);
        pts.push_back({
            a.x * (1.0 - t) + b.x * t,
            a.y * (1.0 - t) + b.y * t,
        });
    }

    return pts;
}

std::vector<Vec2> boundary_points_at(const Semicircle& sc, double radius, int n = kInitialBoundarySamples) {
    const int n_arc = n / 2;
    const int n_flat = n - n_arc;

    std::vector<Vec2> pts;
    pts.reserve(static_cast<std::size_t>(n));

    for (int i = 0; i < n_arc; ++i) {
        const double t = n_arc <= 1 ? 0.0 : static_cast<double>(i) / static_cast<double>(n_arc - 1);
        const double angle = sc.theta - kPi * 0.5 + kPi * t;
        pts.push_back({sc.x + radius * std::cos(angle), sc.y + radius * std::sin(angle)});
    }

    const Vec2 a = endpoint_a_at(sc, radius);
    const Vec2 b = endpoint_b_at(sc, radius);
    for (int i = 0; i < n_flat; ++i) {
        const double t = n_flat <= 1 ? 0.0 : static_cast<double>(i) / static_cast<double>(n_flat - 1);
        pts.push_back({
            a.x * (1.0 - t) + b.x * t,
            a.y * (1.0 - t) + b.y * t,
        });
    }

    return pts;
}

Vec2 farthest_boundary_point_from(const Semicircle& sc, const Vec2& q) {
    std::array<Vec2, 3> candidates{};
    int count = 0;

    const double optimal_angle = std::atan2(sc.y - q.y, sc.x - q.x);
    if (angle_on_arc(optimal_angle, sc.theta)) {
        candidates[count++] = {sc.x + std::cos(optimal_angle), sc.y + std::sin(optimal_angle)};
    }

    candidates[count++] = endpoint_a(sc);
    candidates[count++] = endpoint_b(sc);

    Vec2 best = candidates[0];
    double best_dist_sq = sqr(best.x - q.x) + sqr(best.y - q.y);
    for (int i = 1; i < count; ++i) {
        const double dist_sq = sqr(candidates[i].x - q.x) + sqr(candidates[i].y - q.y);
        if (dist_sq > best_dist_sq) {
            best = candidates[i];
            best_dist_sq = dist_sq;
        }
    }
    return best;
}

Vec2 farthest_boundary_point_from_at(const Semicircle& sc, const Vec2& q, double radius) {
    std::array<Vec2, 3> candidates{};
    int count = 0;

    const double optimal_angle = std::atan2(sc.y - q.y, sc.x - q.x);
    if (angle_on_arc(optimal_angle, sc.theta)) {
        candidates[count++] = {sc.x + radius * std::cos(optimal_angle), sc.y + radius * std::sin(optimal_angle)};
    }

    candidates[count++] = endpoint_a_at(sc, radius);
    candidates[count++] = endpoint_b_at(sc, radius);

    Vec2 best = candidates[0];
    double best_dist_sq = sqr(best.x - q.x) + sqr(best.y - q.y);
    for (int i = 1; i < count; ++i) {
        const double dist_sq = sqr(candidates[i].x - q.x) + sqr(candidates[i].y - q.y);
        if (dist_sq > best_dist_sq) {
            best = candidates[i];
            best_dist_sq = dist_sq;
        }
    }
    return best;
}

std::vector<Vec2> build_polygon(const Semicircle& sc, int arc_points) {
    std::vector<Vec2> poly;
    poly.reserve(static_cast<std::size_t>(arc_points));
    const double start = sc.theta - kPi * 0.5;
    const double step = kPi / static_cast<double>(arc_points - 1);
    for (int i = 0; i < arc_points; ++i) {
        const double angle = start + step * static_cast<double>(i);
        poly.push_back({sc.x + std::cos(angle), sc.y + std::sin(angle)});
    }
    return poly;
}

std::vector<Vec2> build_polygon_at(const Semicircle& sc, int arc_points, double radius) {
    std::vector<Vec2> poly;
    poly.reserve(static_cast<std::size_t>(arc_points));
    const double start = sc.theta - kPi * 0.5;
    const double step = kPi / static_cast<double>(arc_points - 1);
    for (int i = 0; i < arc_points; ++i) {
        const double angle = start + step * static_cast<double>(i);
        poly.push_back({sc.x + radius * std::cos(angle), sc.y + radius * std::sin(angle)});
    }
    return poly;
}

double polygon_area(const std::vector<Vec2>& poly) {
    if (poly.size() < 3) {
        return 0.0;
    }
    double area = 0.0;
    for (std::size_t i = 0; i < poly.size(); ++i) {
        const Vec2& a = poly[i];
        const Vec2& b = poly[(i + 1) % poly.size()];
        area += cross(a, b);
    }
    return std::abs(area) * 0.5;
}

bool inside_half_plane(const Vec2& p, const Vec2& a, const Vec2& b) {
    return cross(b - a, p - a) >= -1e-12;
}

Vec2 line_intersection(const Vec2& s, const Vec2& e, const Vec2& a, const Vec2& b) {
    const Vec2 se = e - s;
    const Vec2 ab = b - a;
    const double denom = cross(se, ab);
    if (std::abs(denom) < 1e-15) {
        return {(s.x + e.x) * 0.5, (s.y + e.y) * 0.5};
    }
    const double t = cross(a - s, ab) / denom;
    return s + se * t;
}

double convex_intersection_area(const std::vector<Vec2>& subject, const std::vector<Vec2>& clip) {
    if (subject.empty() || clip.empty()) {
        return 0.0;
    }

    std::vector<Vec2> output = subject;
    for (std::size_t i = 0; i < clip.size(); ++i) {
        const Vec2& a = clip[i];
        const Vec2& b = clip[(i + 1) % clip.size()];
        if (output.empty()) {
            return 0.0;
        }

        std::vector<Vec2> input = std::move(output);
        output.clear();
        const Vec2* prev = &input.back();
        bool prev_inside = inside_half_plane(*prev, a, b);
        for (const Vec2& cur : input) {
            const bool cur_inside = inside_half_plane(cur, a, b);
            if (cur_inside) {
                if (!prev_inside) {
                    output.push_back(line_intersection(*prev, cur, a, b));
                }
                output.push_back(cur);
            } else if (prev_inside) {
                output.push_back(line_intersection(*prev, cur, a, b));
            }
            prev = &cur;
            prev_inside = cur_inside;
        }
    }
    return polygon_area(output);
}

double shape_radius_at(const Semicircle& sc, const Vec2& center) {
    const Vec2 far = farthest_boundary_point_from(sc, center);
    return distance(far, center);
}

double shape_radius_at(const Semicircle& sc, const Vec2& center, double radius) {
    const Vec2 far = farthest_boundary_point_from_at(sc, center, radius);
    return distance(far, center);
}

Circle compute_mec(const Layout& layout) {
    std::vector<Vec2> pts;
    pts.reserve(static_cast<std::size_t>(kNumSemicircles * kInitialBoundarySamples));
    for (const Semicircle& sc : layout) {
        const std::vector<Vec2> boundary = boundary_points(sc);
        pts.insert(pts.end(), boundary.begin(), boundary.end());
    }

    Circle mec = minimum_enclosing_circle(pts);
    for (int iter = 0; iter < 20; ++iter) {
        bool changed = false;
        for (const Semicircle& sc : layout) {
            const Vec2 p = farthest_boundary_point_from(sc, {mec.x, mec.y});
            if (distance(p, {mec.x, mec.y}) > mec.r + 1e-12) {
                pts.push_back(p);
                changed = true;
            }
        }
        if (!changed) {
            break;
        }
        mec = minimum_enclosing_circle(pts);
    }
    return mec;
}

Circle compute_mec_at_radius(const Layout& layout, double radius) {
    std::vector<Vec2> pts;
    pts.reserve(static_cast<std::size_t>(kNumSemicircles * kInitialBoundarySamples));
    for (const Semicircle& sc : layout) {
        const std::vector<Vec2> boundary = boundary_points_at(sc, radius);
        pts.insert(pts.end(), boundary.begin(), boundary.end());
    }

    Circle mec = minimum_enclosing_circle(pts);
    for (int iter = 0; iter < 20; ++iter) {
        bool changed = false;
        for (const Semicircle& sc : layout) {
            const Vec2 p = farthest_boundary_point_from_at(sc, {mec.x, mec.y}, radius);
            if (distance(p, {mec.x, mec.y}) > mec.r + 1e-12) {
                pts.push_back(p);
                changed = true;
            }
        }
        if (!changed) {
            break;
        }
        mec = minimum_enclosing_circle(pts);
    }
    return mec;
}

RadiusEval exact_radius_to_origin_gradient(const Semicircle& sc) {
    const double ux = std::cos(sc.theta);
    const double uy = std::sin(sc.theta);
    const double dot_value = sc.x * ux + sc.y * uy;

    if (dot_value >= 0.0) {
        const double d = std::hypot(sc.x, sc.y);
        RadiusEval eval;
        eval.radius = d + 1.0;
        if (d > 1e-12) {
            eval.grad_x = sc.x / d;
            eval.grad_y = sc.y / d;
        }
        return eval;
    }

    const double perp = -sc.x * uy + sc.y * ux;
    const double s = perp >= 0.0 ? 1.0 : -1.0;
    const double ex = sc.x - s * uy;
    const double ey = sc.y + s * ux;
    const double r = std::hypot(ex, ey);

    RadiusEval eval;
    eval.radius = r;
    if (r > 1e-12) {
        eval.grad_x = ex / r;
        eval.grad_y = ey / r;
        eval.grad_theta = -s * dot_value / r;
    }
    return eval;
}

int automatic_mode_a(double phi, double theta) {
    return (std::cos(phi - theta) >= 0.0) ? 0 : ((std::sin(phi - theta) >= 0.0) ? 1 : -1);
}

int automatic_mode_b(double phi, double theta) {
    return (std::cos(phi - theta) <= 0.0) ? 0 : ((std::sin(phi - theta) >= 0.0) ? 1 : -1);
}

double eval_sep_phi(const Semicircle& a, const Semicircle& b, double phi, int mode_a, int mode_b) {
    if (mode_a == 99) {
        mode_a = automatic_mode_a(phi, a.theta);
    }
    if (mode_b == 99) {
        mode_b = automatic_mode_b(phi, b.theta);
    }

    double value = (b.x - a.x) * std::cos(phi) + (b.y - a.y) * std::sin(phi);
    value -= (mode_a == 0) ? 1.0 : static_cast<double>(mode_a) * std::sin(phi - a.theta);
    value -= (mode_b == 0) ? 1.0 : static_cast<double>(mode_b) * std::sin(phi - b.theta);
    return value;
}

double eval_sep_phi_at(
    const Semicircle& a,
    const Semicircle& b,
    double phi,
    int mode_a,
    int mode_b,
    double radius
) {
    if (mode_a == 99) {
        mode_a = automatic_mode_a(phi, a.theta);
    }
    if (mode_b == 99) {
        mode_b = automatic_mode_b(phi, b.theta);
    }

    double value = (b.x - a.x) * std::cos(phi) + (b.y - a.y) * std::sin(phi);
    value -= radius * ((mode_a == 0) ? 1.0 : static_cast<double>(mode_a) * std::sin(phi - a.theta));
    value -= radius * ((mode_b == 0) ? 1.0 : static_cast<double>(mode_b) * std::sin(phi - b.theta));
    return value;
}

SepInfo separation_margin(const Semicircle& a, const Semicircle& b) {
    std::array<double, 6> breaks{{
        mod_two_pi(a.theta - kPi * 0.5),
        mod_two_pi(a.theta + kPi * 0.5),
        mod_two_pi(a.theta + kPi),
        mod_two_pi(b.theta - kPi * 0.5),
        mod_two_pi(b.theta + kPi * 0.5),
        mod_two_pi(b.theta),
    }};
    std::sort(breaks.begin(), breaks.end());

    std::vector<double> uniq;
    uniq.reserve(breaks.size());
    for (double value : breaks) {
        if (uniq.empty() || std::abs(value - uniq.back()) > 1e-12) {
            uniq.push_back(value);
        }
    }
    if (uniq.empty()) {
        uniq.push_back(0.0);
    }

    std::vector<double> ext = uniq;
    ext.push_back(uniq.front() + kTwoPi);

    SepInfo best;
    best.margin = -std::numeric_limits<double>::infinity();

    auto consider = [&](double phi, int mode_a, int mode_b) {
        const double margin = eval_sep_phi(a, b, phi, mode_a, mode_b);
        if (margin > best.margin) {
            best.margin = margin;
            best.phi = mod_two_pi(phi);
            best.mode_a = mode_a == 99 ? automatic_mode_a(phi, a.theta) : mode_a;
            best.mode_b = mode_b == 99 ? automatic_mode_b(phi, b.theta) : mode_b;
        }
    };

    for (std::size_t i = 0; i < uniq.size(); ++i) {
        const double left = ext[i];
        const double right = ext[i + 1];
        const double mid = 0.5 * (left + right);
        const int mode_a = automatic_mode_a(mid, a.theta);
        const int mode_b = automatic_mode_b(mid, b.theta);

        double A = b.x - a.x;
        double B = b.y - a.y;
        if (mode_a != 0) {
            A += static_cast<double>(mode_a) * std::sin(a.theta);
            B += -static_cast<double>(mode_a) * std::cos(a.theta);
        }
        if (mode_b != 0) {
            A += static_cast<double>(mode_b) * std::sin(b.theta);
            B += -static_cast<double>(mode_b) * std::cos(b.theta);
        }

        const double psi = std::atan2(B, A);
        consider(left, 99, 99);
        consider(right, 99, 99);
        for (double candidate : {psi, psi + kTwoPi, psi - kTwoPi}) {
            if (candidate > left && candidate < right) {
                consider(candidate, mode_a, mode_b);
            }
        }
    }

    for (double value : uniq) {
        consider(value, 99, 99);
    }
    return best;
}

SepInfo separation_margin_at(const Semicircle& a, const Semicircle& b, double radius) {
    std::array<double, 6> breaks{{
        mod_two_pi(a.theta - kPi * 0.5),
        mod_two_pi(a.theta + kPi * 0.5),
        mod_two_pi(a.theta + kPi),
        mod_two_pi(b.theta - kPi * 0.5),
        mod_two_pi(b.theta + kPi * 0.5),
        mod_two_pi(b.theta),
    }};
    std::sort(breaks.begin(), breaks.end());

    std::vector<double> uniq;
    uniq.reserve(breaks.size());
    for (double value : breaks) {
        if (uniq.empty() || std::abs(value - uniq.back()) > 1e-12) {
            uniq.push_back(value);
        }
    }
    if (uniq.empty()) {
        uniq.push_back(0.0);
    }

    std::vector<double> ext = uniq;
    ext.push_back(uniq.front() + kTwoPi);

    SepInfo best;
    best.margin = -std::numeric_limits<double>::infinity();

    auto consider = [&](double phi, int mode_a, int mode_b) {
        const double margin = eval_sep_phi_at(a, b, phi, mode_a, mode_b, radius);
        if (margin > best.margin) {
            best.margin = margin;
            best.phi = mod_two_pi(phi);
            best.mode_a = mode_a == 99 ? automatic_mode_a(phi, a.theta) : mode_a;
            best.mode_b = mode_b == 99 ? automatic_mode_b(phi, b.theta) : mode_b;
        }
    };

    for (std::size_t i = 0; i < uniq.size(); ++i) {
        const double left = ext[i];
        const double right = ext[i + 1];
        const double mid = 0.5 * (left + right);
        const int mode_a = automatic_mode_a(mid, a.theta);
        const int mode_b = automatic_mode_b(mid, b.theta);

        double A = b.x - a.x;
        double B = b.y - a.y;
        if (mode_a != 0) {
            A += radius * static_cast<double>(mode_a) * std::sin(a.theta);
            B += -radius * static_cast<double>(mode_a) * std::cos(a.theta);
        }
        if (mode_b != 0) {
            A += radius * static_cast<double>(mode_b) * std::sin(b.theta);
            B += -radius * static_cast<double>(mode_b) * std::cos(b.theta);
        }

        const double psi = std::atan2(B, A);
        consider(left, 99, 99);
        consider(right, 99, 99);
        for (double candidate : {psi, psi + kTwoPi, psi - kTwoPi}) {
            if (candidate > left && candidate < right) {
                consider(candidate, mode_a, mode_b);
            }
        }
    }

    for (double value : uniq) {
        consider(value, 99, 99);
    }
    return best;
}

double stable_softplus_scaled(double x, double beta) {
    if (beta <= 0.0) {
        return std::max(0.0, x);
    }
    const double z = x / beta;
    if (z > 40.0) {
        return x;
    }
    if (z < -40.0) {
        return beta * std::exp(z);
    }
    return beta * std::log1p(std::exp(z));
}

double stable_sigmoid(double z) {
    if (z > 40.0) {
        return 1.0;
    }
    if (z < -40.0) {
        return 0.0;
    }
    return 1.0 / (1.0 + std::exp(-z));
}

HomotopyEval evaluate_homotopy(const Layout& layout, const HomotopyPhase& phase) {
    HomotopyEval eval;
    eval.mec = compute_mec_at_radius(layout, phase.radius);
    eval.mec_radius = eval.mec.r;

    std::array<std::vector<Vec2>, kNumSemicircles> polygons;
    for (int i = 0; i < kNumSemicircles; ++i) {
        polygons[i] = build_polygon_at(layout[i], phase.arc_points, phase.radius);
    }

    for (int i = 0; i < kNumSemicircles; ++i) {
        for (int j = i + 1; j < kNumSemicircles; ++j) {
            if (distance({layout[i].x, layout[i].y}, {layout[j].x, layout[j].y}) <= 2.0 * phase.radius + 1e-6) {
                eval.overlap += convex_intersection_area(polygons[i], polygons[j]);
            }
            const double margin = separation_margin_at(layout[i], layout[j], phase.radius).margin;
            eval.min_margin = std::min(eval.min_margin, margin);
            const double slack = phase.margin_target - margin;
            const double s = stable_softplus_scaled(slack, phase.beta);
            eval.margin_penalty += s * s;
        }
    }

    const double center_penalty = phase.center_weight * (sqr(eval.mec.x) + sqr(eval.mec.y));
    eval.cost = eval.mec_radius +
                phase.overlap_weight * eval.overlap +
                phase.margin_weight * eval.margin_penalty +
                center_penalty;
    return eval;
}

Layout run_radius_homotopy_search(
    Layout layout,
    std::mt19937_64& rng,
    std::chrono::steady_clock::time_point deadline
) {
    const std::array<HomotopyPhase, 6> phases{{
        {0.90, 120.0, 10.0, 8e-4, -2.0e-3, 0.10, 220, 0.070, 0.140, 0.020, 0.06, 0.015, 40},
        {0.94, 180.0, 16.0, 5e-4, -1.0e-3, 0.08, 240, 0.050, 0.100, 0.015, 0.03, 0.008, 40},
        {0.97, 260.0, 26.0, 3e-4, -5.0e-4, 0.06, 260, 0.032, 0.070, 0.012, 0.015, 0.004, 48},
        {0.985, 420.0, 42.0, 2e-4, -2.0e-4, 0.05, 300, 0.020, 0.050, 0.009, 0.008, 0.002, 56},
        {0.995, 700.0, 70.0, 1e-4, -7.0e-5, 0.04, 340, 0.012, 0.035, 0.007, 0.004, 0.001, 64},
        {1.000, 1100.0, 120.0, 7e-5, -2.0e-5, 0.03, 420, 0.007, 0.022, 0.005, 0.002, 0.0003, 72},
    }};

    const Circle start_mec = compute_mec(layout);
    layout = scale_about_point(layout, {start_mec.x, start_mec.y}, 0.90);

    std::uniform_real_distribution<double> coin(0.0, 1.0);
    std::uniform_int_distribution<int> pick_index(0, kNumSemicircles - 1);

    for (std::size_t stage_idx = 0; stage_idx < phases.size(); ++stage_idx) {
        if (search_should_stop(deadline)) {
            break;
        }

        const HomotopyPhase& phase = phases[stage_idx];
        HomotopyEval current = evaluate_homotopy(layout, phase);
        Layout best_layout = layout;
        HomotopyEval best_eval = current;

        std::normal_distribution<double> move_xy(0.0, phase.step_xy);
        std::normal_distribution<double> move_theta(0.0, phase.step_theta);
        std::normal_distribution<double> scale_noise(0.0, phase.scale_step);
        std::normal_distribution<double> rotate_noise(0.0, phase.step_theta * 0.5);

        for (int iter = 0; iter < phase.iterations; ++iter) {
            if (search_should_stop(deadline)) {
                break;
            }

            const double t = static_cast<double>(iter) / static_cast<double>(std::max(1, phase.iterations - 1));
            const double temperature = phase.start_temp + (phase.end_temp - phase.start_temp) * t;

            Layout next = layout;
            const double op = coin(rng);
            if (op < 0.08) {
                const double scale = std::clamp(1.0 + scale_noise(rng), 0.96, 1.03);
                next = scale_about_point(next, {current.mec.x, current.mec.y}, scale);
            } else if (op < 0.14) {
                const double angle = rotate_noise(rng);
                next = rotate_layout(next, angle);
            } else if (op < 0.20) {
                const double shift_x = -0.45 * current.mec.x + move_xy(rng) * 0.25;
                const double shift_y = -0.45 * current.mec.y + move_xy(rng) * 0.25;
                for (Semicircle& sc : next) {
                    sc.x += shift_x;
                    sc.y += shift_y;
                }
            } else {
                const int idx = pick_index(rng);
                const double dx = current.mec.x - next[idx].x;
                const double dy = current.mec.y - next[idx].y;
                const double d = std::hypot(dx, dy);
                const Vec2 radial = d > 1e-12 ? Vec2{dx / d, dy / d} : Vec2{1.0, 0.0};
                const Vec2 tangent{-radial.y, radial.x};

                if (coin(rng) < 0.90) {
                    next[idx].x += move_xy(rng);
                    next[idx].y += move_xy(rng);
                }
                if (coin(rng) < 0.60) {
                    next[idx].x += radial.x * move_xy(rng);
                    next[idx].y += radial.y * move_xy(rng);
                }
                if (coin(rng) < 0.45) {
                    next[idx].x += tangent.x * move_xy(rng);
                    next[idx].y += tangent.y * move_xy(rng);
                }
                if (coin(rng) < 0.78) {
                    next[idx].theta = normalize_angle(next[idx].theta + move_theta(rng));
                }
                if (coin(rng) < 0.10) {
                    next[idx].theta = std::atan2(radial.y, radial.x);
                }
            }

            const HomotopyEval candidate = evaluate_homotopy(next, phase);
            const double delta = candidate.cost - current.cost;
            const bool accept =
                delta <= 0.0 ||
                (temperature > 0.0 && coin(rng) < std::exp(-delta / std::max(1e-12, temperature)));
            if (!accept) {
                continue;
            }

            layout = next;
            current = candidate;
            if (current.cost + 1e-12 < best_eval.cost) {
                best_layout = layout;
                best_eval = current;
            }
        }

        layout = best_layout;
        if (stage_idx + 1 < phases.size()) {
            const double next_radius = phases[stage_idx + 1].radius;
            const double factor =
                std::clamp((phase.radius / next_radius) * (0.998 - 0.006 * coin(rng)), 0.94, 1.002);
            layout = scale_about_point(layout, {best_eval.mec.x, best_eval.mec.y}, factor);
        }
    }

    return layout;
}

ObjectiveSummary objective_value(const ParamVector& x, const ObjectivePhase& phase) {
    const Layout layout = to_layout(x);

    std::array<double, kNumSemicircles> radii{};
    double exact_radius = 0.0;
    double max_radius = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < kNumSemicircles; ++i) {
        radii[i] = exact_radius_to_origin_gradient(layout[i]).radius;
        exact_radius = std::max(exact_radius, radii[i]);
        max_radius = std::max(max_radius, radii[i]);
    }

    double smooth_radius = max_radius;
    if (phase.tau > 0.0) {
        double sum_exp = 0.0;
        for (double radius : radii) {
            sum_exp += std::exp((radius - max_radius) / phase.tau);
        }
        smooth_radius = max_radius + phase.tau * std::log(sum_exp);
    }

    double min_margin = std::numeric_limits<double>::infinity();
    double penalty = 0.0;
    for (int i = 0; i < kNumSemicircles; ++i) {
        for (int j = i + 1; j < kNumSemicircles; ++j) {
            const double margin = separation_margin(layout[i], layout[j]).margin;
            min_margin = std::min(min_margin, margin);
            const double slack = phase.margin_target - margin;
            const double s = stable_softplus_scaled(slack, phase.beta);
            penalty += s * s;
        }
    }

    return {smooth_radius + phase.lambda * penalty, exact_radius, min_margin};
}

ObjectiveEval objective_and_gradient(const ParamVector& x, const ObjectivePhase& phase) {
    const Layout layout = to_layout(x);
    ObjectiveEval eval;

    std::array<RadiusEval, kNumSemicircles> radius_evals{};
    std::array<double, kNumSemicircles> radius_weights{};

    double max_radius = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < kNumSemicircles; ++i) {
        radius_evals[i] = exact_radius_to_origin_gradient(layout[i]);
        eval.exact_radius = std::max(eval.exact_radius, radius_evals[i].radius);
        max_radius = std::max(max_radius, radius_evals[i].radius);
    }

    double smooth_radius = max_radius;
    if (phase.tau > 0.0) {
        double sum_exp = 0.0;
        for (int i = 0; i < kNumSemicircles; ++i) {
            radius_weights[i] = std::exp((radius_evals[i].radius - max_radius) / phase.tau);
            sum_exp += radius_weights[i];
        }
        smooth_radius = max_radius + phase.tau * std::log(sum_exp);
        for (double& weight : radius_weights) {
            weight /= sum_exp;
        }
    } else {
        for (int i = 0; i < kNumSemicircles; ++i) {
            radius_weights[i] = radius_evals[i].radius + 1e-12 >= max_radius ? 1.0 : 0.0;
        }
    }

    eval.cost = smooth_radius;
    for (int i = 0; i < kNumSemicircles; ++i) {
        eval.grad[3 * i + 0] += radius_weights[i] * radius_evals[i].grad_x;
        eval.grad[3 * i + 1] += radius_weights[i] * radius_evals[i].grad_y;
        eval.grad[3 * i + 2] += radius_weights[i] * radius_evals[i].grad_theta;
    }

    for (int i = 0; i < kNumSemicircles; ++i) {
        for (int j = i + 1; j < kNumSemicircles; ++j) {
            const SepInfo sep = separation_margin(layout[i], layout[j]);
            eval.min_margin = std::min(eval.min_margin, sep.margin);

            const double slack = phase.margin_target - sep.margin;
            const double s = stable_softplus_scaled(slack, phase.beta);
            if (s <= 0.0) {
                continue;
            }

            const double sigmoid = phase.beta <= 0.0
                ? (slack > 0.0 ? 1.0 : 0.0)
                : stable_sigmoid(slack / phase.beta);
            const double coeff = -2.0 * phase.lambda * s * sigmoid;
            eval.cost += phase.lambda * s * s;

            const double cos_phi = std::cos(sep.phi);
            const double sin_phi = std::sin(sep.phi);
            eval.grad[3 * i + 0] += coeff * (-cos_phi);
            eval.grad[3 * i + 1] += coeff * (-sin_phi);
            eval.grad[3 * j + 0] += coeff * cos_phi;
            eval.grad[3 * j + 1] += coeff * sin_phi;

            if (sep.mode_a != 0) {
                eval.grad[3 * i + 2] += coeff * (static_cast<double>(sep.mode_a) * std::cos(sep.phi - layout[i].theta));
            }
            if (sep.mode_b != 0) {
                eval.grad[3 * j + 2] += coeff * (static_cast<double>(sep.mode_b) * std::cos(sep.phi - layout[j].theta));
            }
        }
    }

    return eval;
}

void normalize_fivefold(FivefoldVector& p) {
    p[0] = std::clamp(p[0], 0.2, 3.2);
    p[2] = std::clamp(p[2], 0.2, 3.2);
    p[5] = std::clamp(p[5], 0.2, 3.2);
    p[1] = normalize_angle(p[1]);
    p[3] = normalize_angle(p[3]);
    p[4] = normalize_angle(p[4]);
    p[6] = normalize_angle(p[6]);
    p[7] = normalize_angle(p[7]);
}

ObjectiveSummary fivefold_objective_value(const FivefoldVector& p, const ObjectivePhase& phase) {
    const Layout layout = fivefold_layout_from_params(p);
    return objective_value(to_vector(layout), phase);
}

FivefoldVector fivefold_gradient(FivefoldVector p, const ObjectivePhase& phase) {
    FivefoldVector grad{};
    for (int i = 0; i < kFivefoldDim; ++i) {
        const bool is_radius = (i == 0 || i == 2 || i == 5);
        const double h = (is_radius ? 1e-5 : 1e-5) * (1.0 + std::abs(p[i]));

        FivefoldVector plus = p;
        FivefoldVector minus = p;
        plus[i] += h;
        minus[i] -= h;
        normalize_fivefold(plus);
        normalize_fivefold(minus);

        const double fp = fivefold_objective_value(plus, phase).cost;
        const double fm = fivefold_objective_value(minus, phase).cost;
        grad[i] = (fp - fm) / (2.0 * h);
    }
    return grad;
}

FivefoldVector lbfgs_direction(
    const FivefoldVector& grad,
    const std::vector<FivefoldVector>& S,
    const std::vector<FivefoldVector>& Y,
    const std::vector<double>& rho
) {
    const int m = static_cast<int>(S.size());
    FivefoldVector q = grad;
    std::vector<double> alpha(static_cast<std::size_t>(m), 0.0);

    for (int i = m - 1; i >= 0; --i) {
        alpha[static_cast<std::size_t>(i)] = rho[static_cast<std::size_t>(i)] * dot_product(S[static_cast<std::size_t>(i)], q);
        axpy(q, -alpha[static_cast<std::size_t>(i)], Y[static_cast<std::size_t>(i)]);
    }

    double gamma = 1.0;
    if (m > 0) {
        const double sy = dot_product(S.back(), Y.back());
        const double yy = dot_product(Y.back(), Y.back());
        if (yy > 1e-20) {
            gamma = sy / yy;
        }
    }

    FivefoldVector r = scaled(q, gamma);
    for (int i = 0; i < m; ++i) {
        const double beta = rho[static_cast<std::size_t>(i)] * dot_product(Y[static_cast<std::size_t>(i)], r);
        axpy(r, alpha[static_cast<std::size_t>(i)] - beta, S[static_cast<std::size_t>(i)]);
    }

    for (double& value : r) {
        value = -value;
    }
    return r;
}

FivefoldVector optimize_fivefold(
    FivefoldVector p,
    const std::vector<ObjectivePhase>& phases,
    std::chrono::steady_clock::time_point deadline
) {
    normalize_fivefold(p);

    constexpr std::size_t kMemoryLimit = 8;
    for (const ObjectivePhase& phase : phases) {
        std::vector<FivefoldVector> S;
        std::vector<FivefoldVector> Y;
        std::vector<double> rho;
        ObjectiveSummary value = fivefold_objective_value(p, phase);
        FivefoldVector grad = fivefold_gradient(p, phase);

        const int max_iters = std::max(12, phase.max_iters / 2);
        for (int iter = 0; iter < max_iters; ++iter) {
            if (search_should_stop(deadline)) {
                return p;
            }

            FivefoldVector direction = lbfgs_direction(grad, S, Y, rho);
            double gtp = dot_product(grad, direction);
            if (!(gtp < -1e-12) || !std::isfinite(gtp)) {
                direction = scaled(grad, -1.0);
                gtp = -dot_product(grad, grad);
            }

            const double dir_norm = l2_norm(direction);
            if (!std::isfinite(dir_norm) || dir_norm < 1e-14) {
                break;
            }

            if (dir_norm > 0.25) {
                direction = scaled(direction, 0.25 / dir_norm);
                gtp = dot_product(grad, direction);
            }

            double step = 1.0;
            bool accepted = false;
            FivefoldVector p_new{};
            ObjectiveSummary value_new{};
            for (int line_search = 0; line_search < 20; ++line_search) {
                p_new = add_scaled(p, direction, step);
                normalize_fivefold(p_new);
                value_new = fivefold_objective_value(p_new, phase);
                if (std::isfinite(value_new.cost) &&
                    value_new.cost <= value.cost + 1e-4 * step * gtp) {
                    accepted = true;
                    break;
                }
                step *= 0.5;
            }
            if (!accepted) {
                break;
            }

            const FivefoldVector grad_new = fivefold_gradient(p_new, phase);
            const FivefoldVector s = subtract(p_new, p);
            const FivefoldVector y = subtract(grad_new, grad);
            const double sy = dot_product(s, y);
            if (sy > 1e-12 && std::isfinite(sy)) {
                if (S.size() == kMemoryLimit) {
                    S.erase(S.begin());
                    Y.erase(Y.begin());
                    rho.erase(rho.begin());
                }
                S.push_back(s);
                Y.push_back(y);
                rho.push_back(1.0 / sy);
            }

            p = p_new;
            value = value_new;
            grad = grad_new;
            if (l2_norm(grad) < 1e-7) {
                break;
            }
        }
    }

    return p;
}

ObjectiveSummary orbit_break_objective_value(
    const FivefoldVector& p,
    const OrbitBreakVector& q,
    const ObjectivePhase& phase
) {
    const Layout layout = orbit_break_layout_from_params(p, q);
    return objective_value(to_vector(layout), phase);
}

OrbitBreakEval orbit_break_objective_and_gradient(
    const FivefoldVector& p,
    const OrbitBreakVector& q,
    const ObjectivePhase& phase
) {
    const Layout layout = orbit_break_layout_from_params(p, q);
    const ObjectiveEval full = objective_and_gradient(to_vector(layout), phase);

    OrbitBreakEval eval;
    eval.cost = full.cost;
    eval.exact_radius = full.exact_radius;
    eval.min_margin = full.min_margin;

    for (int k = 0; k < 5; ++k) {
        const double base = kTwoPi * static_cast<double>(k) / 5.0;
        const auto basis = orbit_break_basis(k);

        for (int orbit = 0; orbit < kOrbitBreakOrbits; ++orbit) {
            const int q_base = orbit * kOrbitBreakPerOrbit;
            const int idx = 3 * k + 1 + orbit;
            const double orbit_r = orbit == 0 ? p[2] : p[5];
            const double orbit_angle = orbit == 0 ? p[3] : p[6];
            const double dr = orbit_break_combo(q, q_base + 0, basis);
            const double da = orbit_break_combo(q, q_base + 4, basis);
            const double radius = orbit_r + dr;
            const double angle = base + orbit_angle + da;
            const double cos_angle = std::cos(angle);
            const double sin_angle = std::sin(angle);

            const double gx = full.grad[3 * idx + 0];
            const double gy = full.grad[3 * idx + 1];
            const double gt = full.grad[3 * idx + 2];

            for (int h = 0; h < kOrbitBreakHarmonics; ++h) {
                const double wave = basis[h];
                eval.grad[q_base + 0 + h] += gx * wave * cos_angle + gy * wave * sin_angle;
                eval.grad[q_base + 4 + h] += gx * (-radius * wave * sin_angle) + gy * (radius * wave * cos_angle);
                eval.grad[q_base + 8 + h] += gt * wave;
            }
        }
    }

    return eval;
}

OrbitBreakVector lbfgs_direction(
    const OrbitBreakVector& grad,
    const std::vector<OrbitBreakVector>& S,
    const std::vector<OrbitBreakVector>& Y,
    const std::vector<double>& rho
) {
    const int m = static_cast<int>(S.size());
    OrbitBreakVector q = grad;
    std::vector<double> alpha(static_cast<std::size_t>(m), 0.0);

    for (int i = m - 1; i >= 0; --i) {
        alpha[static_cast<std::size_t>(i)] = rho[static_cast<std::size_t>(i)] * dot_product(S[static_cast<std::size_t>(i)], q);
        axpy(q, -alpha[static_cast<std::size_t>(i)], Y[static_cast<std::size_t>(i)]);
    }

    double gamma = 1.0;
    if (m > 0) {
        const double sy = dot_product(S.back(), Y.back());
        const double yy = dot_product(Y.back(), Y.back());
        if (yy > 1e-20) {
            gamma = sy / yy;
        }
    }

    OrbitBreakVector r = scaled(q, gamma);
    for (int i = 0; i < m; ++i) {
        const double beta = rho[static_cast<std::size_t>(i)] * dot_product(Y[static_cast<std::size_t>(i)], r);
        axpy(r, alpha[static_cast<std::size_t>(i)] - beta, S[static_cast<std::size_t>(i)]);
    }

    for (double& value : r) {
        value = -value;
    }
    return r;
}

OrbitBreakVector optimize_orbit_break(
    const FivefoldVector& p,
    OrbitBreakVector q,
    const std::vector<ObjectivePhase>& phases,
    std::chrono::steady_clock::time_point deadline
) {
    normalize_orbit_break(q);

    constexpr std::size_t kMemoryLimit = 8;
    for (const ObjectivePhase& phase : phases) {
        std::vector<OrbitBreakVector> S;
        std::vector<OrbitBreakVector> Y;
        std::vector<double> rho;
        OrbitBreakEval eval = orbit_break_objective_and_gradient(p, q, phase);

        const int max_iters = std::max(16, phase.max_iters / 2);
        for (int iter = 0; iter < max_iters; ++iter) {
            if (search_should_stop(deadline)) {
                return q;
            }

            OrbitBreakVector direction = lbfgs_direction(eval.grad, S, Y, rho);
            double gtp = dot_product(eval.grad, direction);
            if (!(gtp < -1e-12) || !std::isfinite(gtp)) {
                direction = scaled(eval.grad, -1.0);
                gtp = -dot_product(eval.grad, eval.grad);
            }

            const double dir_norm = l2_norm(direction);
            if (!std::isfinite(dir_norm) || dir_norm < 1e-14) {
                break;
            }

            if (dir_norm > 0.18) {
                direction = scaled(direction, 0.18 / dir_norm);
                gtp = dot_product(eval.grad, direction);
            }

            double step = 1.0;
            bool accepted = false;
            OrbitBreakVector q_new{};
            ObjectiveSummary value_new{};
            for (int line_search = 0; line_search < 20; ++line_search) {
                q_new = add_scaled(q, direction, step);
                normalize_orbit_break(q_new);
                value_new = orbit_break_objective_value(p, q_new, phase);
                if (std::isfinite(value_new.cost) &&
                    value_new.cost <= eval.cost + 1e-4 * step * gtp) {
                    accepted = true;
                    break;
                }
                step *= 0.5;
            }
            if (!accepted) {
                break;
            }

            const OrbitBreakEval eval_new = orbit_break_objective_and_gradient(p, q_new, phase);
            const OrbitBreakVector s = subtract(q_new, q);
            const OrbitBreakVector y = subtract(eval_new.grad, eval.grad);
            const double sy = dot_product(s, y);
            if (sy > 1e-12 && std::isfinite(sy)) {
                if (S.size() == kMemoryLimit) {
                    S.erase(S.begin());
                    Y.erase(Y.begin());
                    rho.erase(rho.begin());
                }
                S.push_back(s);
                Y.push_back(y);
                rho.push_back(1.0 / sy);
            }

            q = q_new;
            eval = eval_new;
            if (l2_norm(eval.grad) < 1e-7) {
                break;
            }
        }
    }

    return q;
}

ParamVector lbfgs_direction(
    const ParamVector& grad,
    const std::vector<ParamVector>& S,
    const std::vector<ParamVector>& Y,
    const std::vector<double>& rho
) {
    const int m = static_cast<int>(S.size());
    ParamVector q = grad;
    std::vector<double> alpha(static_cast<std::size_t>(m), 0.0);

    for (int i = m - 1; i >= 0; --i) {
        alpha[static_cast<std::size_t>(i)] = rho[static_cast<std::size_t>(i)] * dot_product(S[static_cast<std::size_t>(i)], q);
        axpy(q, -alpha[static_cast<std::size_t>(i)], Y[static_cast<std::size_t>(i)]);
    }

    double gamma = 1.0;
    if (m > 0) {
        const double sy = dot_product(S.back(), Y.back());
        const double yy = dot_product(Y.back(), Y.back());
        if (yy > 1e-20) {
            gamma = sy / yy;
        }
    }

    ParamVector r = scaled(q, gamma);
    for (int i = 0; i < m; ++i) {
        const double beta = rho[static_cast<std::size_t>(i)] * dot_product(Y[static_cast<std::size_t>(i)], r);
        axpy(r, alpha[static_cast<std::size_t>(i)] - beta, S[static_cast<std::size_t>(i)]);
    }

    for (double& value : r) {
        value = -value;
    }
    return r;
}

void center_by_mec(ParamVector& x) {
    const Layout layout = to_layout(x);
    const Circle mec = compute_mec(layout);
    for (int i = 0; i < kNumSemicircles; ++i) {
        x[3 * i + 0] -= mec.x;
        x[3 * i + 1] -= mec.y;
    }
}

ParamVector optimize_layout(
    ParamVector x,
    const std::vector<ObjectivePhase>& phases,
    std::chrono::steady_clock::time_point deadline
) {
    normalize_angles(x);
    center_by_mec(x);

    constexpr std::size_t kMemoryLimit = 12;
    for (const ObjectivePhase& phase : phases) {
        std::vector<ParamVector> S;
        std::vector<ParamVector> Y;
        std::vector<double> rho;
        ObjectiveEval eval = objective_and_gradient(x, phase);

        for (int iter = 0; iter < phase.max_iters; ++iter) {
            if (search_should_stop(deadline)) {
                return x;
            }

            ParamVector direction = lbfgs_direction(eval.grad, S, Y, rho);
            double gtp = dot_product(eval.grad, direction);
            if (!(gtp < -1e-12) || !std::isfinite(gtp)) {
                direction = scaled(eval.grad, -1.0);
                gtp = -dot_product(eval.grad, eval.grad);
            }

            const double dir_norm = l2_norm(direction);
            if (!std::isfinite(dir_norm) || dir_norm < 1e-14) {
                break;
            }

            const double max_step_norm = 0.35;
            if (dir_norm > max_step_norm) {
                direction = scaled(direction, max_step_norm / dir_norm);
                gtp = dot_product(eval.grad, direction);
            }

            double step = 1.0;
            bool accepted = false;
            ParamVector x_new{};
            ObjectiveSummary summary_new{};

            for (int line_search = 0; line_search < 20; ++line_search) {
                x_new = add_scaled(x, direction, step);
                normalize_angles(x_new);
                summary_new = objective_value(x_new, phase);
                if (std::isfinite(summary_new.cost) &&
                    summary_new.cost <= eval.cost + 1e-4 * step * gtp) {
                    accepted = true;
                    break;
                }
                step *= 0.5;
            }

            if (!accepted) {
                break;
            }

            if ((iter + 1) % std::max(1, phase.recenter_period) == 0) {
                center_by_mec(x_new);
                normalize_angles(x_new);
                summary_new = objective_value(x_new, phase);
                S.clear();
                Y.clear();
                rho.clear();
            }

            const ObjectiveEval eval_new = objective_and_gradient(x_new, phase);
            ParamVector s = subtract(x_new, x);
            ParamVector y = subtract(eval_new.grad, eval.grad);
            const double sy = dot_product(s, y);
            if (sy > 1e-12 && std::isfinite(sy)) {
                if (S.size() == kMemoryLimit) {
                    S.erase(S.begin());
                    Y.erase(Y.begin());
                    rho.erase(rho.begin());
                }
                S.push_back(s);
                Y.push_back(y);
                rho.push_back(1.0 / sy);
            }

            x = x_new;
            eval = eval_new;
            if (l2_norm(eval.grad) < 1e-7) {
                break;
            }
        }

        center_by_mec(x);
        normalize_angles(x);
    }

    return x;
}

Layout rotate_layout(Layout layout, double angle) {
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    for (Semicircle& sc : layout) {
        const double x = sc.x;
        const double y = sc.y;
        sc.x = c * x - s * y;
        sc.y = s * x + c * y;
        sc.theta = normalize_angle(sc.theta + angle);
    }
    return layout;
}

Layout mirror_layout_x(Layout layout) {
    for (Semicircle& sc : layout) {
        sc.x = -sc.x;
        sc.theta = normalize_angle(kPi - sc.theta);
    }
    return layout;
}

Layout scale_about_point(Layout layout, const Vec2& center, double scale) {
    for (Semicircle& sc : layout) {
        sc.x = center.x + (sc.x - center.x) * scale;
        sc.y = center.y + (sc.y - center.y) * scale;
        sc.theta = normalize_angle(sc.theta);
    }
    return layout;
}

Layout low_frequency_distort_layout(
    Layout layout,
    std::mt19937_64& rng,
    double amp_xy,
    double amp_theta
) {
    std::normal_distribution<double> noise(0.0, 1.0);
    struct Mode {
        double ax = 0.0;
        double ay = 0.0;
        double at = 0.0;
        double phase = 0.0;
        int k = 1;
    };

    std::array<Mode, 4> modes{};
    for (int i = 0; i < 4; ++i) {
        modes[static_cast<std::size_t>(i)] = {
            amp_xy * noise(rng),
            amp_xy * noise(rng),
            amp_theta * noise(rng),
            kTwoPi * (0.5 + 0.5 * noise(rng)),
            i + 1,
        };
    }

    for (Semicircle& sc : layout) {
        const double angle = std::atan2(sc.y, sc.x);
        double dx = 0.0;
        double dy = 0.0;
        double dt = 0.0;
        for (const Mode& mode : modes) {
            const double wave = static_cast<double>(mode.k) * angle + mode.phase;
            dx += mode.ax * std::cos(wave);
            dy += mode.ay * std::sin(wave);
            dt += mode.at * std::sin(wave);
        }
        sc.x += dx;
        sc.y += dy;
        sc.theta = normalize_angle(sc.theta + dt);
    }

    return layout;
}

std::vector<ObjectivePhase> scaled_phases(const std::vector<ObjectivePhase>& phases, double scale) {
    std::vector<ObjectivePhase> out = phases;
    for (ObjectivePhase& phase : out) {
        phase.max_iters = std::max(4, static_cast<int>(std::lround(static_cast<double>(phase.max_iters) * scale)));
    }
    return out;
}

Layout jitter_layout(
    Layout layout,
    std::mt19937_64& rng,
    double xy_sigma,
    double theta_sigma,
    double global_scale_sigma
) {
    std::normal_distribution<double> move_xy(0.0, xy_sigma);
    std::normal_distribution<double> move_theta(0.0, theta_sigma);
    std::normal_distribution<double> scale_noise(0.0, global_scale_sigma);
    std::uniform_real_distribution<double> coin(0.0, 1.0);
    std::uniform_real_distribution<double> rotate(-kPi, kPi);

    layout = rotate_layout(layout, rotate(rng));
    if (coin(rng) < 0.5) {
        layout = mirror_layout_x(layout);
    }

    const double scale = std::clamp(1.0 + scale_noise(rng), 0.96, 1.04);
    layout = scale_about_point(layout, {0.0, 0.0}, scale);

    for (Semicircle& sc : layout) {
        sc.x += move_xy(rng);
        sc.y += move_xy(rng);
        sc.theta = normalize_angle(sc.theta + move_theta(rng));
    }

    return layout;
}

Layout compress_best_seed(const Layout& layout, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> factor(0.9850, 1.0025);
    std::uniform_real_distribution<double> sx_dist(0.985, 1.015);
    std::uniform_real_distribution<double> sy_dist(0.985, 1.015);
    const Circle mec = compute_mec(layout);
    Layout out = scale_about_point(layout, {mec.x, mec.y}, factor(rng));
    const double sx = sx_dist(rng);
    const double sy = sy_dist(rng);
    std::normal_distribution<double> move_xy(0.0, 0.010);
    std::normal_distribution<double> move_theta(0.0, 0.018);
    for (Semicircle& sc : out) {
        sc.x = mec.x + (sc.x - mec.x) * sx;
        sc.y = mec.y + (sc.y - mec.y) * sy;
        sc.x += move_xy(rng);
        sc.y += move_xy(rng);
        sc.theta = normalize_angle(sc.theta + move_theta(rng));
    }
    return out;
}

ValidationSummary validate_layout(Layout layout) {
    layout = round_layout(layout);

    ValidationSummary summary;
    summary.valid = true;
    summary.min_margin = std::numeric_limits<double>::infinity();
    std::array<std::vector<Vec2>, kNumSemicircles> polygons;
    for (int i = 0; i < kNumSemicircles; ++i) {
        polygons[i] = build_polygon(layout[i], kValidateArcPoints);
    }
    for (int i = 0; i < kNumSemicircles; ++i) {
        for (int j = i + 1; j < kNumSemicircles; ++j) {
            const double margin = separation_margin(layout[i], layout[j]).margin;
            summary.min_margin = std::min(summary.min_margin, margin);
            if (distance({layout[i].x, layout[i].y}, {layout[j].x, layout[j].y}) <= 2.000001) {
                const double area = convex_intersection_area(polygons[i], polygons[j]);
                if (area > kOverlapTol) {
                    summary.valid = false;
                }
            }
            if (margin < kValidMarginTol - 0.01) {
                summary.valid = false;
            }
        }
    }

    summary.mec = compute_mec(layout);
    summary.score = summary.mec.r;

    for (const Semicircle& sc : layout) {
        const Vec2 p = farthest_boundary_point_from(sc, {summary.mec.x, summary.mec.y});
        if (distance(p, {summary.mec.x, summary.mec.y}) > summary.mec.r + kContainTol) {
            summary.valid = false;
        }
    }

    return summary;
}

void try_scale_polish(
    Layout& layout,
    double& best_score,
    std::chrono::steady_clock::time_point deadline
) {
    const Circle mec = compute_mec(layout);
    double low = 0.90;
    double high = 1.0;
    Layout best_layout = layout;

    for (int iter = 0; iter < 24; ++iter) {
        if (search_should_stop(deadline)) {
            break;
        }

        const double mid = 0.5 * (low + high);
        Layout candidate = scale_about_point(layout, {mec.x, mec.y}, mid);
        const ValidationSummary val = validate_layout(candidate);
        if (val.valid && val.score + 1e-12 < best_score) {
            high = mid;
            best_layout = round_layout(candidate);
            best_score = val.score;
        } else {
            low = mid;
        }
    }

    layout = best_layout;
}

void micro_coordinate_polish(
    Layout& layout,
    double& best_score,
    std::chrono::steady_clock::time_point deadline
) {
    struct StepCfg {
        double xy;
        double theta;
    };
    const std::array<StepCfg, 5> steps{{
        {0.0015, 0.0030},
        {0.0008, 0.0016},
        {0.0004, 0.0008},
        {0.0002, 0.0004},
        {0.0001, 0.0002},
    }};

    for (const StepCfg& step : steps) {
        bool improved = true;
        while (improved && !search_should_stop(deadline)) {
            improved = false;
            const Circle mec = compute_mec(layout);

            std::vector<int> order(kNumSemicircles);
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
                return shape_radius_at(layout[lhs], {mec.x, mec.y}) > shape_radius_at(layout[rhs], {mec.x, mec.y});
            });

            for (int idx : order) {
                if (search_should_stop(deadline)) {
                    return;
                }

                const double dx = mec.x - layout[idx].x;
                const double dy = mec.y - layout[idx].y;
                const double d = std::hypot(dx, dy);
                const Vec2 radial = d > 1e-12 ? Vec2{dx / d, dy / d} : Vec2{1.0, 0.0};
                const Vec2 tangent{-radial.y, radial.x};
                const double inward_angle = std::atan2(radial.y, radial.x);

                std::vector<Semicircle> candidates;
                candidates.reserve(14);
                candidates.push_back({layout[idx].x + step.xy, layout[idx].y, layout[idx].theta});
                candidates.push_back({layout[idx].x - step.xy, layout[idx].y, layout[idx].theta});
                candidates.push_back({layout[idx].x, layout[idx].y + step.xy, layout[idx].theta});
                candidates.push_back({layout[idx].x, layout[idx].y - step.xy, layout[idx].theta});
                candidates.push_back({
                    layout[idx].x + radial.x * step.xy,
                    layout[idx].y + radial.y * step.xy,
                    layout[idx].theta,
                });
                candidates.push_back({
                    layout[idx].x - radial.x * step.xy,
                    layout[idx].y - radial.y * step.xy,
                    layout[idx].theta,
                });
                candidates.push_back({
                    layout[idx].x + tangent.x * step.xy,
                    layout[idx].y + tangent.y * step.xy,
                    layout[idx].theta,
                });
                candidates.push_back({
                    layout[idx].x - tangent.x * step.xy,
                    layout[idx].y - tangent.y * step.xy,
                    layout[idx].theta,
                });
                candidates.push_back({layout[idx].x, layout[idx].y, normalize_angle(layout[idx].theta + step.theta)});
                candidates.push_back({layout[idx].x, layout[idx].y, normalize_angle(layout[idx].theta - step.theta)});
                candidates.push_back({layout[idx].x, layout[idx].y, inward_angle});
                candidates.push_back({layout[idx].x, layout[idx].y, normalize_angle(inward_angle + step.theta)});
                candidates.push_back({layout[idx].x, layout[idx].y, normalize_angle(inward_angle - step.theta)});

                for (const Semicircle& candidate : candidates) {
                    Layout next = layout;
                    next[idx] = rounded_shape(candidate);
                    const ValidationSummary val = validate_layout(next);
                    if (!val.valid || val.score + 1e-12 >= best_score) {
                        continue;
                    }
                    layout = round_layout(next);
                    best_score = val.score;
                    improved = true;
                    break;
                }
                if (improved) {
                    break;
                }
            }
        }
    }
}

Layout choose_seed(
    std::mt19937_64& rng,
    SharedBest& shared,
    const std::vector<Layout>& base_seeds
) {
    std::uniform_real_distribution<double> coin(0.0, 1.0);

    if (auto best = snapshot_best(shared); best.has_value()) {
        const double p = coin(rng);
        if (p < 0.30) {
            return jitter_layout(*best, rng, 0.004, 0.008, 0.0015);
        }
        if (p < 0.82) {
            return compress_best_seed(*best, rng);
        }
        if (p < 0.92) {
            return jitter_layout(*best, rng, 0.018, 0.036, 0.004);
        }
    }

    if (coin(rng) < 0.12) {
        return random_shell_seed(rng);
    }

    std::uniform_int_distribution<int> pick(0, static_cast<int>(base_seeds.size() - 1));
    const Layout base = base_seeds[static_cast<std::size_t>(pick(rng))];
    const double p = coin(rng);
    if (p < 0.50) {
        return jitter_layout(base, rng, 0.008, 0.016, 0.002);
    }
    if (p < 0.85) {
        return jitter_layout(base, rng, 0.030, 0.060, 0.006);
    }
    return jitter_layout(base, rng, 0.080, 0.160, 0.014);
}

bool strict_refine_and_publish(
    SharedBest& shared,
    Layout seed_layout,
    int thread_id,
    StrategyId strategy_id,
    const std::string& strategy_label,
    const std::vector<ObjectivePhase>& phases,
    const std::vector<ObjectivePhase>& repair_phases,
    std::chrono::steady_clock::time_point deadline
) {
    ParamVector x = optimize_layout(to_vector(seed_layout), phases, deadline);
    Layout layout = to_layout(x);
    ValidationSummary val = validate_layout(layout);
    if (!val.valid && !search_should_stop(deadline)) {
        x = optimize_layout(x, repair_phases, deadline);
        layout = to_layout(x);
        val = validate_layout(layout);
    }
    if (!val.valid) {
        return false;
    }

    layout = round_layout(layout);
    double best_score = val.score;
    try_scale_polish(layout, best_score, deadline);
    micro_coordinate_polish(layout, best_score, deadline);
    const ValidationSummary polished = validate_layout(layout);
    if (!polished.valid) {
        return false;
    }

    return maybe_publish_valid(shared, layout, polished, thread_id, strategy_id, strategy_label);
}

bool record_relaxed_then_repair(
    SharedBest& shared,
    Layout layout,
    int thread_id,
    StrategyId strategy_id,
    const std::string& strategy_label,
    const ObjectivePhase& relaxed_archive_phase,
    const std::vector<ObjectivePhase>& phases,
    const std::vector<ObjectivePhase>& repair_phases,
    std::chrono::steady_clock::time_point deadline
) {
    const ObjectiveSummary relaxed = objective_value(to_vector(layout), relaxed_archive_phase);
    bool inserted = maybe_record_relaxed_candidate(
        shared,
        layout,
        relaxed.cost,
        relaxed.min_margin,
        thread_id,
        strategy_id,
        strategy_label
    );

    if (inserted && !search_should_stop(deadline)) {
        inserted = strict_refine_and_publish(
            shared,
            layout,
            thread_id,
            strategy_id,
            strategy_label,
            phases,
            repair_phases,
            deadline
        ) || inserted;
    }

    return inserted;
}

void worker_loop(
    int thread_id,
    SharedBest& shared,
    std::chrono::steady_clock::time_point deadline,
    const std::vector<Layout>& base_seeds,
    const std::vector<ObjectivePhase>& phases,
    const std::vector<ObjectivePhase>& repair_phases,
    const std::vector<ObjectivePhase>& fivefold_phases,
    const std::vector<ObjectivePhase>& orbit_break_phases,
    const std::vector<ObjectivePhase>& ring_phases,
    const std::vector<ObjectivePhase>& relaxed_explore_phases,
    const ObjectivePhase& relaxed_archive_phase
) {
    std::mt19937_64 rng(
        static_cast<std::uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) ^
        (0x9E3779B97F4A7C15ULL * static_cast<std::uint64_t>(thread_id + 1))
    );
    std::uniform_real_distribution<double> coin(0.0, 1.0);

    std::array<int, kStrategyCount> no_insert_streak{};
    std::array<double, kStrategyCount> budget_scale{};
    budget_scale.fill(1.0);
    std::size_t strategy_cursor = static_cast<std::size_t>(thread_id % kStrategyCount);
    std::size_t ring_cursor = static_cast<std::size_t>(thread_id);

    while (!search_should_stop(deadline)) {
        const StrategyId strategy = static_cast<StrategyId>(strategy_cursor % kStrategyCount);
        strategy_cursor += 1;
        const int strategy_index = static_cast<int>(strategy);
        note_strategy_attempt(shared, strategy);

        const double budget = budget_scale[static_cast<std::size_t>(strategy_index)];
        const std::vector<ObjectivePhase> strict_phases = scaled_phases(phases, budget);
        const std::vector<ObjectivePhase> repair_scaled = scaled_phases(repair_phases, budget);
        const std::vector<ObjectivePhase> fivefold_scaled = scaled_phases(fivefold_phases, budget);
        const std::vector<ObjectivePhase> orbit_scaled = scaled_phases(orbit_break_phases, budget);
        const std::vector<ObjectivePhase> ring_scaled = scaled_phases(ring_phases, budget);
        const std::vector<ObjectivePhase> relaxed_scaled = scaled_phases(relaxed_explore_phases, budget);

        bool inserted_any = false;
        std::string strategy_label = strategy_name(strategy);

        if (strategy == StrategyId::StrictArchiveRefine) {
            std::vector<CandidateRecord> official_archive = snapshot_official_archive(shared);
            Layout seed = choose_seed(rng, shared, base_seeds);
            if (!official_archive.empty()) {
                const std::size_t limit = std::min<std::size_t>(official_archive.size(), 8);
                std::uniform_int_distribution<int> pick(0, static_cast<int>(limit - 1));
                seed = official_archive[static_cast<std::size_t>(pick(rng))].layout;
            } else if (auto best = snapshot_best(shared); best.has_value()) {
                seed = *best;
            }
            if (coin(rng) < 0.55) {
                seed = compress_best_seed(seed, rng);
            } else {
                seed = jitter_layout(seed, rng, 0.008, 0.016, 0.002);
            }
            inserted_any = strict_refine_and_publish(
                shared,
                seed,
                thread_id,
                strategy,
                strategy_label,
                strict_phases,
                repair_scaled,
                deadline
            );
        } else if (strategy == StrategyId::FivefoldExact) {
            FivefoldVector p = fivefold_params();
            std::normal_distribution<double> dr(0.0, 0.08);
            std::normal_distribution<double> da(0.0, 0.08);
            p[0] += dr(rng);
            p[2] += dr(rng);
            p[5] += dr(rng);
            p[1] += da(rng);
            p[3] += da(rng);
            p[4] += da(rng);
            p[6] += da(rng);
            p[7] += da(rng);
            normalize_fivefold(p);
            p = optimize_fivefold(p, fivefold_scaled, deadline);
            inserted_any = strict_refine_and_publish(
                shared,
                fivefold_layout_from_params(p),
                thread_id,
                strategy,
                strategy_label,
                strict_phases,
                repair_scaled,
                deadline
            );
        } else if (strategy == StrategyId::FivefoldBroken) {
            FivefoldVector p = fivefold_params();
            std::normal_distribution<double> dr(0.0, 0.06);
            std::normal_distribution<double> da(0.0, 0.06);
            p[0] += dr(rng);
            p[2] += dr(rng);
            p[5] += dr(rng);
            p[1] += da(rng);
            p[3] += da(rng);
            p[4] += da(rng);
            p[6] += da(rng);
            p[7] += da(rng);
            normalize_fivefold(p);
            p = optimize_fivefold(p, fivefold_scaled, deadline);

            OrbitBreakVector q{};
            std::normal_distribution<double> q_radius(0.0, 0.010);
            std::normal_distribution<double> q_angle(0.0, 0.015);
            std::normal_distribution<double> q_theta(0.0, 0.025);
            for (int orbit = 0; orbit < kOrbitBreakOrbits; ++orbit) {
                const int q_base = orbit * kOrbitBreakPerOrbit;
                for (int i = 0; i < kOrbitBreakHarmonics; ++i) {
                    q[q_base + 0 + i] = q_radius(rng);
                    q[q_base + 4 + i] = q_angle(rng);
                    q[q_base + 8 + i] = q_theta(rng);
                }
            }
            normalize_orbit_break(q);
            q = optimize_orbit_break(p, q, orbit_scaled, deadline);
            inserted_any = strict_refine_and_publish(
                shared,
                orbit_break_layout_from_params(p, q),
                thread_id,
                strategy,
                strategy_label,
                strict_phases,
                repair_scaled,
                deadline
            );
        } else if (strategy == StrategyId::RingPartition) {
            const auto& specs = ring_family_specs();
            const RingFamilySpec& spec = specs[ring_cursor % specs.size()];
            const RingOrientationPreset preset = static_cast<RingOrientationPreset>((ring_cursor / specs.size()) % 3);
            ring_cursor += 1;
            strategy_label += "/";
            strategy_label += spec.name;
            strategy_label += "/";
            strategy_label += ring_orientation_name(preset);

            std::vector<double> params = ring_family_initial_params(spec, preset, rng);
            auto normalizer = [&](std::vector<double>& values) {
                normalize_ring_family_params(values, spec);
            };
            auto builder = [&](const std::vector<double>& values) {
                return ring_family_layout_from_params(spec, preset, values);
            };
            params = optimize_reduced(params, ring_scaled, deadline, builder, normalizer);
            inserted_any = record_relaxed_then_repair(
                shared,
                builder(params),
                thread_id,
                strategy,
                strategy_label,
                relaxed_archive_phase,
                strict_phases,
                repair_scaled,
                deadline
            );
        } else if (strategy == StrategyId::CompressedContinuation) {
            Layout seed = choose_seed(rng, shared, base_seeds);
            std::vector<CandidateRecord> official_archive = snapshot_official_archive(shared);
            std::vector<CandidateRecord> relaxed_archive = snapshot_relaxed_archive(shared);
            if (!official_archive.empty() && coin(rng) < 0.65) {
                const std::size_t limit = std::min<std::size_t>(official_archive.size(), 10);
                std::uniform_int_distribution<int> pick(0, static_cast<int>(limit - 1));
                seed = official_archive[static_cast<std::size_t>(pick(rng))].layout;
            } else if (!relaxed_archive.empty() && coin(rng) < 0.35) {
                const std::size_t limit = std::min<std::size_t>(relaxed_archive.size(), 10);
                std::uniform_int_distribution<int> pick(0, static_cast<int>(limit - 1));
                seed = relaxed_archive[static_cast<std::size_t>(pick(rng))].layout;
            }
            seed = compress_best_seed(seed, rng);
            seed = run_radius_homotopy_search(seed, rng, deadline);
            inserted_any = strict_refine_and_publish(
                shared,
                seed,
                thread_id,
                strategy,
                strategy_label,
                strict_phases,
                repair_scaled,
                deadline
            );
        } else {
            Layout seed = choose_seed(rng, shared, base_seeds);
            std::vector<CandidateRecord> relaxed_archive = snapshot_relaxed_archive(shared);
            if (!relaxed_archive.empty()) {
                const std::size_t limit = std::min<std::size_t>(relaxed_archive.size(), 12);
                std::uniform_int_distribution<int> pick(0, static_cast<int>(limit - 1));
                seed = relaxed_archive[static_cast<std::size_t>(pick(rng))].layout;
            }
            seed = jitter_layout(seed, rng, 0.028, 0.050, 0.006);
            seed = low_frequency_distort_layout(seed, rng, 0.020, 0.040);
            ParamVector relaxed_x = optimize_layout(to_vector(seed), relaxed_scaled, deadline);
            inserted_any = record_relaxed_then_repair(
                shared,
                to_layout(relaxed_x),
                thread_id,
                strategy,
                strategy_label,
                relaxed_archive_phase,
                strict_phases,
                repair_scaled,
                deadline
            );
        }

        if (inserted_any) {
            no_insert_streak[static_cast<std::size_t>(strategy_index)] = 0;
        } else {
            no_insert_streak[static_cast<std::size_t>(strategy_index)] += 1;
            if (no_insert_streak[static_cast<std::size_t>(strategy_index)] >= 3) {
                budget_scale[static_cast<std::size_t>(strategy_index)] =
                    std::max(0.25, budget_scale[static_cast<std::size_t>(strategy_index)] * 0.5);
                no_insert_streak[static_cast<std::size_t>(strategy_index)] = 0;
            }
        }
    }
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if ((arg == "--seconds" || arg == "--search-seconds") && i + 1 < argc) {
            options.seconds = std::max(0.0, std::stod(argv[++i]));
            if (options.seconds == 0.0) {
                options.run_until_interrupt = true;
            }
        } else if (arg == "--threads" && i + 1 < argc) {
            options.threads = std::max(1, std::stoi(argv[++i]));
        } else if (arg == "--forever" || arg == "--until-interrupt") {
            options.run_until_interrupt = true;
        } else if (arg == "--checkpoint-seconds" && i + 1 < argc) {
            options.checkpoint_seconds = std::max(1, std::stoi(argv[++i]));
        } else if (arg == "--archive-size" && i + 1 < argc) {
            options.archive_size = static_cast<std::size_t>(std::max(1, std::stoi(argv[++i])));
        } else if (arg == "--relaxed-archive-size" && i + 1 < argc) {
            options.relaxed_archive_size = static_cast<std::size_t>(std::max(1, std::stoi(argv[++i])));
        } else if (arg == "--no-resume") {
            options.no_resume = true;
        } else if (arg == "--input" && i + 1 < argc) {
            options.input_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            options.output_path = argv[++i];
        } else if ((arg == "--history" || arg == "--strategy-log") && i + 1 < argc) {
            options.history_path = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: ./solve [--seconds N | --forever] [--threads N] [--checkpoint-seconds N]"
                << " [--archive-size N] [--relaxed-archive-size N] [--no-resume]"
                << " [--input FILE] [--output FILE] [--strategy-log FILE]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options options = parse_args(argc, argv);
        g_stop_requested.store(false, std::memory_order_relaxed);
        std::signal(SIGINT, handle_stop_signal);
#ifdef SIGTERM
        std::signal(SIGTERM, handle_stop_signal);
#endif

        const int hw_threads = static_cast<int>(std::thread::hardware_concurrency());
        const int threads = options.threads > 0 ? options.threads : std::max(1, hw_threads);
        const bool run_until_interrupt = options.run_until_interrupt;
        const std::string official_archive_path = sidecar_path(options.output_path, ".archive.json");
        const std::string relaxed_archive_path = sidecar_path(options.output_path, ".relaxed_archive.json");
        if (options.history_path.empty()) {
            options.history_path = sidecar_path(options.output_path, ".history.tsv");
        }

        std::vector<Layout> base_seeds;
        base_seeds.push_back(known_best_solution());
        base_seeds.push_back(legacy_solution());
        base_seeds.push_back(fivefold_seed());

        if (!options.input_path.empty()) {
            if (auto input = load_solution_json(options.input_path); input.has_value()) {
                base_seeds.push_back(*input);
            }
        } else if (auto input = load_solution_json("solution.json"); input.has_value()) {
            base_seeds.push_back(*input);
        }

        SharedBest shared;
        shared.start_time = std::chrono::steady_clock::now();
        shared.history_path = options.history_path;
        shared.official_archive_path = official_archive_path;
        shared.relaxed_archive_path = relaxed_archive_path;
        shared.archive_size = options.archive_size;
        shared.relaxed_archive_size = options.relaxed_archive_size;
        initialize_history_log(shared.history_path, options.no_resume);

        if (!options.no_resume) {
            if (auto resumed = load_solution_json(options.output_path); resumed.has_value()) {
                base_seeds.push_back(*resumed);
            }
            if (auto archive = load_candidate_archive_json(official_archive_path); archive.has_value()) {
                for (CandidateRecord entry : *archive) {
                    const ValidationSummary val = validate_layout(entry.layout);
                    if (!val.valid) {
                        continue;
                    }
                    entry.layout = round_layout(entry.layout);
                    entry.score = val.score;
                    entry.min_margin = val.min_margin;
                    insert_candidate_archive(shared.official_archive, entry, shared.archive_size);
                    if (!shared.has_valid || val.score + 1e-12 < shared.score) {
                        shared.layout = entry.layout;
                        shared.score = val.score;
                        shared.has_valid = true;
                    }
                    base_seeds.push_back(entry.layout);
                }
            }
            if (auto archive = load_candidate_archive_json(relaxed_archive_path); archive.has_value()) {
                for (CandidateRecord entry : *archive) {
                    entry.layout = round_layout(entry.layout);
                    insert_candidate_archive(shared.relaxed_archive, entry, shared.relaxed_archive_size);
                    base_seeds.push_back(entry.layout);
                }
            }
        }

        for (const Layout& seed : base_seeds) {
            const ValidationSummary val = validate_layout(seed);
            if (val.valid) {
                maybe_publish_valid(shared, seed, val, -1, StrategyId::StrictArchiveRefine, "bootstrap");
            }
        }
        if (!shared.has_valid) {
            throw std::runtime_error("No valid initial seed available.");
        }

        const std::vector<ObjectivePhase> phases{{
            {120.0, 0.0300, 0.0600, -6.0e-4, 60, 4},
            {220.0, 0.0150, 0.0300, -4.0e-4, 80, 4},
            {400.0, 0.0080, 0.0150, -2.5e-4, 100, 3},
            {700.0, 0.0040, 0.0070, -1.2e-4, 120, 3},
            {1100.0, 0.0020, 0.0030, -5.0e-5, 120, 2},
            {1600.0, 0.0010, 0.0015, -2.0e-5, 100, 2},
        }};
        const std::vector<ObjectivePhase> repair_phases{{
            {2200.0, 0.0008, 0.0012, -1.0e-5, 80, 1},
            {3200.0, 0.0005, 0.0008, 0.0, 60, 1},
        }};
        const std::vector<ObjectivePhase> fivefold_phases{{
            {60.0, 0.0400, 0.0800, -1.5e-3, 40, 1},
            {120.0, 0.0200, 0.0400, -8.0e-4, 50, 1},
            {240.0, 0.0100, 0.0200, -3.0e-4, 60, 1},
            {500.0, 0.0040, 0.0080, -8.0e-5, 60, 1},
        }};
        const std::vector<ObjectivePhase> orbit_break_phases{{
            {120.0, 0.0200, 0.0400, -7.0e-4, 48, 1},
            {260.0, 0.0100, 0.0200, -3.5e-4, 64, 1},
            {520.0, 0.0040, 0.0080, -1.2e-4, 80, 1},
            {900.0, 0.0020, 0.0030, -4.0e-5, 80, 1},
        }};
        const std::vector<ObjectivePhase> ring_phases{{
            {90.0, 0.0300, 0.0500, -1.8e-3, 40, 1},
            {180.0, 0.0160, 0.0250, -8.0e-4, 52, 1},
            {360.0, 0.0080, 0.0120, -3.0e-4, 64, 1},
            {700.0, 0.0035, 0.0060, -1.0e-4, 72, 1},
        }};
        const std::vector<ObjectivePhase> relaxed_explore_phases{{
            {80.0, 0.0500, 0.0800, -2.5e-3, 36, 2},
            {160.0, 0.0280, 0.0500, -1.4e-3, 48, 2},
            {300.0, 0.0140, 0.0250, -7.0e-4, 60, 2},
        }};
        const ObjectivePhase relaxed_archive_phase{220.0, 0.0200, 0.0300, -1.0e-3, 48, 2};

        const auto deadline = run_until_interrupt
            ? std::chrono::steady_clock::time_point::max()
            : std::chrono::steady_clock::now() +
                  std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                      std::chrono::duration<double>(options.seconds));

        std::cout << "Running " << threads << " thread(s)";
        if (run_until_interrupt) {
            std::cout << " until interrupted";
        } else {
            std::cout << " for " << std::fixed << std::setprecision(1) << options.seconds << "s";
        }
        std::cout << "  checkpoint=" << options.checkpoint_seconds << "s" << std::endl;

        std::atomic<bool> checkpoint_done{false};
        std::thread checkpoint_thread([&]() {
            try {
                const auto interval = std::chrono::seconds(options.checkpoint_seconds);
                auto next_checkpoint = std::chrono::steady_clock::now() + interval;
                while (!checkpoint_done.load(std::memory_order_relaxed)) {
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                    if (checkpoint_done.load(std::memory_order_relaxed)) {
                        break;
                    }
                    const auto now = std::chrono::steady_clock::now();
                    if (now < next_checkpoint) {
                        continue;
                    }
                    const BestSnapshot best = snapshot_best_state(shared);
                    if (best.has_valid) {
                        write_status_report(
                            best,
                            shared,
                            options.output_path,
                            g_stop_requested.load(std::memory_order_relaxed),
                            false
                        );
                        std::cout << "[checkpoint] score "
                                  << std::fixed << std::setprecision(6) << best.score
                                  << " saved to " << options.output_path << std::endl;
                    }
                    next_checkpoint = now + interval;
                }
            } catch (const std::exception& ex) {
                std::cerr << "[checkpoint] " << ex.what() << std::endl;
            }
        });

        std::vector<std::thread> workers;
        workers.reserve(static_cast<std::size_t>(threads));
        for (int thread_id = 0; thread_id < threads; ++thread_id) {
            workers.emplace_back(
                worker_loop,
                thread_id,
                std::ref(shared),
                deadline,
                std::cref(base_seeds),
                std::cref(phases),
                std::cref(repair_phases),
                std::cref(fivefold_phases),
                std::cref(orbit_break_phases),
                std::cref(ring_phases),
                std::cref(relaxed_explore_phases),
                std::cref(relaxed_archive_phase)
            );
        }
        for (std::thread& worker : workers) {
            worker.join();
        }
        checkpoint_done.store(true, std::memory_order_relaxed);
        checkpoint_thread.join();

        const BestSnapshot best = snapshot_best_state(shared);
        if (!best.has_valid) {
            throw std::runtime_error("Search finished without a valid solution.");
        }

        write_status_report(
            best,
            shared,
            options.output_path,
            g_stop_requested.load(std::memory_order_relaxed),
            true
        );
        if (g_stop_requested.load(std::memory_order_relaxed)) {
            std::cout << "Interrupt received. Best-so-far saved." << std::endl;
        }
        std::cout << "Saved best packing to " << options.output_path << "\n";
        std::cout << "Approximate score: " << std::fixed << std::setprecision(6) << best.score << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
