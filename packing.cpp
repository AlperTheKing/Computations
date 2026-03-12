#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <random>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr int N = 15;
constexpr double PI = 3.141592653589793238462643383279502884;
constexpr double TAU = 2.0 * PI;
constexpr double OFFICIAL_OVERLAP_TOL = 1e-6;
constexpr double CONTAINMENT_TOL = 1e-9;
constexpr double HUGE_SCORE = 1e300;

struct Point {
    double x = 0.0;
    double y = 0.0;
};

struct Semicircle {
    double x = 0.0;
    double y = 0.0;
    double theta = 0.0;
};

struct EvalConfig {
    int arc_points = 128;
    int mec_boundary_points = 128;
    double overlap_tol = OFFICIAL_OVERLAP_TOL;
};

struct FastResult {
    double energy = HUGE_SCORE;
    double radius = HUGE_SCORE;
    double overlap_sum = HUGE_SCORE;
    double containment_excess = HUGE_SCORE;
    bool approx_valid = false;
};

struct ExactResult {
    bool valid = false;
    double radius = HUGE_SCORE;
    double overlap_sum = 0.0;
    Point mec_center{};
    std::string error;
};

struct GlobalBest {
    std::mutex mu;
    double score = HUGE_SCORE;
    std::vector<Semicircle> sol;
    uint64_t improvements = 0;
    std::chrono::steady_clock::time_point last_improve_tp{};
};

struct SearchParams {
    std::string input_path = "best_solution.json";
    std::string output_path = "best_solution.json";
    std::string log_path = "best_solution.log";
    int threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    int seconds = 120;
    int report_every = 0;
    uint64_t seed = 1;
    double penalty_weight = 260.0;
    EvalConfig fast_cfg{64, 32, 5e-6};
    EvalConfig exact_cfg{4096, 128, OFFICIAL_OVERLAP_TOL};
    bool score_only = false;
};

struct Logger {
    std::mutex mu;
    std::ofstream file;
};

struct WorkerSnapshot {
    std::mutex mu;
    bool has = false;
    uint64_t iter = 0;
    uint64_t accepts = 0;
    uint64_t exact_checks = 0;
    uint64_t exact_valid = 0;
    double cur_energy = HUGE_SCORE;
    double cur_radius = HUGE_SCORE;
    double cur_overlap = HUGE_SCORE;
    double cur_containment = HUGE_SCORE;
    double local_best_radius = HUGE_SCORE;
    int stagnation = 0;
};

struct WorkerProfile {
    double init_jitter = 0.0;
    double mutation_scale = 1.0;
    double repair_rate = 0.60;
    int repair_iters = 2;
    double compaction_rate = 0.20;
    double compaction_strength = 0.35;
    bool start_from_best = true;
    double exact_margin = 0.004;
};

inline double sqr(double x) { return x * x; }

inline double clampd(double x, double lo, double hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

inline double dist2(const Point& a, const Point& b) {
    return sqr(a.x - b.x) + sqr(a.y - b.y);
}

inline double dist(const Point& a, const Point& b) {
    return std::sqrt(dist2(a, b));
}

inline double cross(const Point& a, const Point& b, const Point& c) {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

inline double wrap_theta(double t) {
    t = std::fmod(t, TAU);
    if (t < 0.0) t += TAU;
    return t;
}

inline double round6(double x) {
    return std::round(x * 1e6) / 1e6;
}

std::string now_str() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
    return std::string(buf);
}

void log_message(Logger& logger, const std::string& msg) {
    std::lock_guard<std::mutex> lock(logger.mu);
    std::cout << msg << '\n';
    std::cout.flush();
    if (logger.file) {
        logger.file << msg << '\n';
        logger.file.flush();
    }
}

double median_of(std::vector<double> v) {
    if (v.empty()) return HUGE_SCORE;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    if ((n & 1ULL) == 1ULL) return v[n / 2];
    return 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

WorkerProfile make_worker_profile(int tid) {
    switch (tid & 3) {
        case 0:
            return WorkerProfile{0.0, 0.25, 0.92, 2, 0.08, 0.12, true, 0.015};
        case 1:
            return WorkerProfile{0.0003, 0.45, 0.85, 2, 0.10, 0.15, true, 0.012};
        case 2:
            return WorkerProfile{0.0015, 0.95, 0.65, 2, 0.14, 0.20, false, 0.007};
        default:
            return WorkerProfile{0.0035, 1.35, 0.45, 3, 0.18, 0.28, false, 0.006};
    }
}

void publish_worker_snapshot(WorkerSnapshot& ws,
                             uint64_t iter,
                             uint64_t accepts,
                             uint64_t exact_checks,
                             uint64_t exact_valid,
                             const FastResult& cur,
                             const FastResult& local_best_eval,
                             int stagnation) {
    std::lock_guard<std::mutex> lock(ws.mu);
    ws.has = true;
    ws.iter = iter;
    ws.accepts = accepts;
    ws.exact_checks = exact_checks;
    ws.exact_valid = exact_valid;
    ws.cur_energy = cur.energy;
    ws.cur_radius = cur.radius;
    ws.cur_overlap = cur.overlap_sum;
    ws.cur_containment = cur.containment_excess;
    ws.local_best_radius = local_best_eval.radius;
    ws.stagnation = stagnation;
}

double polygon_area_signed(const std::vector<Point>& poly) {
    const int n = static_cast<int>(poly.size());
    if (n < 3) return 0.0;
    double a = 0.0;
    for (int i = 0; i < n; ++i) {
        const Point& p = poly[i];
        const Point& q = poly[(i + 1) % n];
        a += p.x * q.y - p.y * q.x;
    }
    return 0.5 * a;
}

double polygon_area(const std::vector<Point>& poly) {
    return std::abs(polygon_area_signed(poly));
}

Point line_intersection(const Point& s,
                        const Point& e,
                        const Point& a,
                        const Point& b) {
    const double A1 = e.y - s.y;
    const double B1 = s.x - e.x;
    const double C1 = A1 * s.x + B1 * s.y;

    const double A2 = b.y - a.y;
    const double B2 = a.x - b.x;
    const double C2 = A2 * a.x + B2 * a.y;

    const double det = A1 * B2 - A2 * B1;
    if (std::abs(det) < 1e-18) return s;
    return Point{(B2 * C1 - B1 * C2) / det, (A1 * C2 - A2 * C1) / det};
}

bool inside_ccw(const Point& p, const Point& a, const Point& b, double eps = 1e-12) {
    return cross(a, b, p) >= -eps;
}

std::vector<Point> convex_clip(std::vector<Point> subject, const std::vector<Point>& clipper) {
    if (subject.empty() || clipper.empty()) return {};

    for (int i = 0, m = static_cast<int>(clipper.size()); i < m; ++i) {
        const Point& A = clipper[i];
        const Point& B = clipper[(i + 1) % m];

        std::vector<Point> out;
        out.reserve(subject.size() + 1);

        for (int j = 0, n = static_cast<int>(subject.size()); j < n; ++j) {
            const Point& S = subject[j];
            const Point& E = subject[(j + 1) % n];
            const bool s_in = inside_ccw(S, A, B);
            const bool e_in = inside_ccw(E, A, B);

            if (s_in && e_in) {
                out.push_back(E);
            } else if (s_in && !e_in) {
                out.push_back(line_intersection(S, E, A, B));
            } else if (!s_in && e_in) {
                out.push_back(line_intersection(S, E, A, B));
                out.push_back(E);
            }
        }
        subject.swap(out);
        if (subject.size() < 3) return {};
    }
    return subject;
}

std::vector<Point> semicircle_polygon(const Semicircle& sc, int n_arc) {
    n_arc = std::max(8, n_arc);
    std::vector<Point> poly;
    poly.reserve(static_cast<size_t>(n_arc));

    const double start = sc.theta - PI / 2.0;
    const double end = sc.theta + PI / 2.0;
    for (int i = 0; i < n_arc; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(n_arc - 1);
        const double a = start + (end - start) * t;
        poly.push_back(Point{sc.x + std::cos(a), sc.y + std::sin(a)});
    }

    if (polygon_area_signed(poly) < 0.0) std::reverse(poly.begin(), poly.end());
    return poly;
}

double overlap_area_polygonized(const Semicircle& a, const Semicircle& b, int arc_points) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    if (dx * dx + dy * dy > 4.0 + 1e-12) return 0.0;

    std::vector<Point> pa = semicircle_polygon(a, arc_points);
    std::vector<Point> pb = semicircle_polygon(b, arc_points);
    std::vector<Point> inter = convex_clip(pa, pb);
    return polygon_area(inter);
}

Point farthest_boundary_point_from(const Semicircle& sc, double qx, double qy) {
    const double dx = sc.x - qx;
    const double dy = sc.y - qy;

    std::vector<Point> cand;
    cand.reserve(3);

    const double optimal = std::atan2(dy, dx);
    const double diff = std::atan2(std::sin(optimal - sc.theta), std::cos(optimal - sc.theta));
    if (-PI / 2.0 <= diff && diff <= PI / 2.0) {
        cand.push_back(Point{sc.x + std::cos(optimal), sc.y + std::sin(optimal)});
    }

    const double a1 = sc.theta - PI / 2.0;
    const double a2 = sc.theta + PI / 2.0;
    cand.push_back(Point{sc.x + std::cos(a1), sc.y + std::sin(a1)});
    cand.push_back(Point{sc.x + std::cos(a2), sc.y + std::sin(a2)});

    Point best = cand.front();
    double best_d2 = dist2(best, Point{qx, qy});
    for (const Point& p : cand) {
        const double d2 = dist2(p, Point{qx, qy});
        if (d2 > best_d2) {
            best = p;
            best_d2 = d2;
        }
    }
    return best;
}

std::vector<Point> semicircle_boundary_points(const Semicircle& sc, int n) {
    n = std::max(8, n);
    const int n_arc = n / 2;
    const int n_flat = n - n_arc;

    std::vector<Point> pts;
    pts.reserve(static_cast<size_t>(n));

    const double start = sc.theta - PI / 2.0;
    const double end = sc.theta + PI / 2.0;
    for (int i = 0; i < n_arc; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(std::max(1, n_arc - 1));
        const double a = start + (end - start) * t;
        pts.push_back(Point{sc.x + std::cos(a), sc.y + std::sin(a)});
    }

    const double perp = sc.theta + PI / 2.0;
    const Point e1{sc.x + std::cos(perp), sc.y + std::sin(perp)};
    const Point e2{sc.x - std::cos(perp), sc.y - std::sin(perp)};
    for (int i = 0; i < n_flat; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(std::max(1, n_flat - 1));
        pts.push_back(Point{e1.x * (1.0 - t) + e2.x * t, e1.y * (1.0 - t) + e2.y * t});
    }
    return pts;
}

struct Circle {
    Point c{};
    double r = 0.0;
};

Circle circle_from_1(const Point& p) {
    return Circle{p, 0.0};
}

Circle circle_from_2(const Point& p1, const Point& p2) {
    return Circle{Point{(p1.x + p2.x) / 2.0, (p1.y + p2.y) / 2.0}, dist(p1, p2) / 2.0};
}

Circle circle_from_3(const Point& p1, const Point& p2, const Point& p3) {
    const double ax = p1.x, ay = p1.y;
    const double bx = p2.x, by = p2.y;
    const double cx = p3.x, cy = p3.y;
    const double d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));

    if (std::abs(d) < 1e-14) {
        Circle c1 = circle_from_2(p1, p2);
        Circle c2 = circle_from_2(p1, p3);
        Circle c3 = circle_from_2(p2, p3);
        Circle best = c1;
        if (c2.r > best.r) best = c2;
        if (c3.r > best.r) best = c3;
        return best;
    }

    const double ux = ((ax * ax + ay * ay) * (by - cy) +
                       (bx * bx + by * by) * (cy - ay) +
                       (cx * cx + cy * cy) * (ay - by)) / d;
    const double uy = ((ax * ax + ay * ay) * (cx - bx) +
                       (bx * bx + by * by) * (ax - cx) +
                       (cx * cx + cy * cy) * (bx - ax)) / d;

    const Point u{ux, uy};
    return Circle{u, dist(u, p1)};
}

bool in_circle(const Circle& c, const Point& p, double eps = 1e-10) {
    return dist(c.c, p) <= c.r + eps;
}

Circle make_circle(const std::vector<Point>& b) {
    if (b.empty()) return Circle{Point{0.0, 0.0}, 0.0};
    if (b.size() == 1) return circle_from_1(b[0]);
    if (b.size() == 2) return circle_from_2(b[0], b[1]);
    return circle_from_3(b[0], b[1], b[2]);
}

Circle minimum_enclosing_circle(std::vector<Point> pts) {
    std::mt19937 rng(42);
    std::shuffle(pts.begin(), pts.end(), rng);

    Circle c = make_circle({});
    for (int i = 0; i < static_cast<int>(pts.size()); ++i) {
        if (!in_circle(c, pts[i])) {
            c = make_circle({pts[i]});
            for (int j = 0; j < i; ++j) {
                if (!in_circle(c, pts[j])) {
                    c = make_circle({pts[i], pts[j]});
                    for (int k = 0; k < j; ++k) {
                        if (!in_circle(c, pts[k])) {
                            c = make_circle({pts[i], pts[j], pts[k]});
                        }
                    }
                }
            }
        }
    }
    return c;
}

Circle compute_mec(const std::vector<Semicircle>& scs, int boundary_points) {
    std::vector<Point> all_pts;
    all_pts.reserve(static_cast<size_t>(scs.size() * static_cast<size_t>(boundary_points + 4)));

    for (const Semicircle& sc : scs) {
        std::vector<Point> pts = semicircle_boundary_points(sc, boundary_points);
        all_pts.insert(all_pts.end(), pts.begin(), pts.end());
    }

    Circle c = minimum_enclosing_circle(all_pts);
    for (int iter = 0; iter < 20; ++iter) {
        std::vector<Point> add;
        add.reserve(scs.size());
        for (const Semicircle& sc : scs) {
            const Point f = farthest_boundary_point_from(sc, c.c.x, c.c.y);
            if (dist(f, c.c) > c.r + 1e-12) add.push_back(f);
        }
        if (add.empty()) break;
        all_pts.insert(all_pts.end(), add.begin(), add.end());
        c = minimum_enclosing_circle(all_pts);
    }
    return c;
}

std::vector<Semicircle> round_solution(const std::vector<Semicircle>& in) {
    std::vector<Semicircle> out = in;
    for (Semicircle& sc : out) {
        sc.x = round6(sc.x);
        sc.y = round6(sc.y);
        sc.theta = round6(wrap_theta(sc.theta));
    }
    return out;
}

FastResult fast_eval(const std::vector<Semicircle>& in,
                     const EvalConfig& cfg,
                     double penalty_weight) {
    FastResult r;

    double overlap_sum = 0.0;
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            overlap_sum += overlap_area_polygonized(in[i], in[j], cfg.arc_points);
        }
    }

    const Circle mec = compute_mec(in, cfg.mec_boundary_points);

    double containment_excess = 0.0;
    for (int i = 0; i < N; ++i) {
        const Point f = farthest_boundary_point_from(in[i], mec.c.x, mec.c.y);
        const double ex = std::max(0.0, dist(f, mec.c) - mec.r);
        containment_excess += ex * ex;
    }

    r.radius = mec.r;
    r.overlap_sum = overlap_sum;
    r.containment_excess = containment_excess;
    r.approx_valid = overlap_sum <= cfg.overlap_tol && containment_excess <= 1e-10;
    r.energy = r.radius + penalty_weight * overlap_sum + 800.0 * penalty_weight * containment_excess;
    return r;
}

ExactResult exact_validate(const std::vector<Semicircle>& in, const EvalConfig& cfg) {
    ExactResult out;
    std::vector<Semicircle> scs = round_solution(in);

    if (static_cast<int>(scs.size()) != N) {
        out.error = "Expected exactly 15 semicircles";
        return out;
    }

    double overlap_sum = 0.0;
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            const double a = overlap_area_polygonized(scs[i], scs[j], cfg.arc_points);
            overlap_sum += a;
            if (a > cfg.overlap_tol) {
                out.error = "Overlap detected";
                out.overlap_sum = overlap_sum;
                return out;
            }
        }
    }

    const Circle mec = compute_mec(scs, cfg.mec_boundary_points);
    for (int i = 0; i < N; ++i) {
        const Point f = farthest_boundary_point_from(scs[i], mec.c.x, mec.c.y);
        if (dist(f, mec.c) > mec.r + CONTAINMENT_TOL) {
            out.error = "Containment failed";
            out.radius = mec.r;
            out.overlap_sum = overlap_sum;
            out.mec_center = mec.c;
            return out;
        }
    }

    out.valid = true;
    out.radius = mec.r;
    out.overlap_sum = overlap_sum;
    out.mec_center = mec.c;
    return out;
}

bool load_solution_json(const std::string& path, std::vector<Semicircle>& out) {
    std::ifstream in(path);
    if (!in) return false;

    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    static const std::regex num_re(R"([+-]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][+-]?\d+)?)");

    std::vector<double> nums;
    nums.reserve(3 * N);
    for (auto it = std::sregex_iterator(text.begin(), text.end(), num_re); it != std::sregex_iterator(); ++it) {
        nums.push_back(std::stod(it->str()));
    }

    if (nums.size() != static_cast<size_t>(3 * N)) return false;

    out.assign(N, Semicircle{});
    for (int i = 0; i < N; ++i) {
        out[i].x = nums[3 * i + 0];
        out[i].y = nums[3 * i + 1];
        out[i].theta = nums[3 * i + 2];
    }
    return true;
}

void save_solution_json(const std::string& path, const std::vector<Semicircle>& scs) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Failed to write output file: " + path);

    out << std::fixed << std::setprecision(6);
    out << "[\n";
    for (size_t i = 0; i < scs.size(); ++i) {
        out << "  {\"x\": " << round6(scs[i].x)
            << ", \"y\": " << round6(scs[i].y)
            << ", \"theta\": " << round6(wrap_theta(scs[i].theta)) << "}";
        if (i + 1 != scs.size()) out << ',';
        out << '\n';
    }
    out << "]\n";
}

void recenter_mean(std::vector<Semicircle>& scs) {
    double mx = 0.0;
    double my = 0.0;
    for (const Semicircle& sc : scs) {
        mx += sc.x;
        my += sc.y;
    }
    mx /= static_cast<double>(scs.size());
    my /= static_cast<double>(scs.size());

    for (Semicircle& sc : scs) {
        sc.x -= mx;
        sc.y -= my;
        sc.theta = wrap_theta(sc.theta);
    }
}

std::vector<double> build_weights(const std::vector<Semicircle>& base) {
    const Circle mec = compute_mec(base, 128);

    std::vector<int> near_counts(N, 0);
    std::vector<double> slack(N, 0.0);

    for (int i = 0; i < N; ++i) {
        const Point f = farthest_boundary_point_from(base[i], mec.c.x, mec.c.y);
        slack[i] = mec.r - dist(f, mec.c);
        for (int j = i + 1; j < N; ++j) {
            const double d = std::hypot(base[i].x - base[j].x, base[i].y - base[j].y);
            if (d < 2.15) {
                near_counts[i]++;
                near_counts[j]++;
            }
        }
    }

    std::vector<double> w(N, 1.0);
    for (int i = 0; i < N; ++i) {
        w[i] += static_cast<double>(near_counts[i]);
        if (slack[i] < 0.003) w[i] += 3.0;
        if (slack[i] < 0.020) w[i] += 1.0;
    }
    return w;
}

std::vector<std::pair<int, int>> build_pair_list(const std::vector<Semicircle>& base) {
    std::vector<std::pair<int, int>> pairs;
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            const double d = std::hypot(base[i].x - base[j].x, base[i].y - base[j].y);
            if (d < 2.25) pairs.emplace_back(i, j);
        }
    }
    if (pairs.empty()) {
        for (int i = 0; i < N; ++i) {
            for (int j = i + 1; j < N; ++j) pairs.emplace_back(i, j);
        }
    }
    return pairs;
}

int weighted_pick(std::mt19937_64& rng, const std::vector<double>& w) {
    std::discrete_distribution<int> dist(w.begin(), w.end());
    return dist(rng);
}

double rand01(std::mt19937_64& rng) {
    return std::uniform_real_distribution<double>(0.0, 1.0)(rng);
}

double gauss(std::mt19937_64& rng, double sigma) {
    std::normal_distribution<double> dist(0.0, sigma);
    return dist(rng);
}

void quick_repair(std::vector<Semicircle>& s,
                  std::mt19937_64& rng,
                  int iters,
                  int overlap_arc,
                  int mec_pts);

double angle_distance(double a, double b) {
    return std::atan2(std::sin(a - b), std::cos(a - b));
}

double best_theta_for_center(const Semicircle& sc, const Point& center) {
    const double radial = std::atan2(center.y - sc.y, center.x - sc.x);
    const std::array<double, 12> candidates{
        wrap_theta(sc.theta),
        wrap_theta(sc.theta + 0.20),
        wrap_theta(sc.theta - 0.20),
        wrap_theta(radial),
        wrap_theta(radial + PI),
        wrap_theta(radial + PI * 0.50),
        wrap_theta(radial - PI * 0.50),
        wrap_theta(radial + PI * 0.25),
        wrap_theta(radial - PI * 0.25),
        wrap_theta(radial + PI * 0.75),
        wrap_theta(radial - PI * 0.75),
        wrap_theta(sc.theta + PI)
    };

    double best_theta = candidates[0];
    double best_dist = HUGE_SCORE;
    for (double theta : candidates) {
        Semicircle probe = sc;
        probe.theta = theta;
        const Point f = farthest_boundary_point_from(probe, center.x, center.y);
        const double d = dist(f, center);
        if (d + 1e-12 < best_dist) {
            best_dist = d;
            best_theta = theta;
        }
    }
    return best_theta;
}

void pressure_compact(std::vector<Semicircle>& s,
                      std::mt19937_64& rng,
                      double strength,
                      int passes) {
    for (int pass = 0; pass < passes; ++pass) {
        const Circle mec = compute_mec(s, 40);
        std::vector<std::pair<double, int>> order;
        order.reserve(N);
        for (int i = 0; i < N; ++i) {
            const Point f = farthest_boundary_point_from(s[i], mec.c.x, mec.c.y);
            const double slack = mec.r - dist(f, mec.c);
            order.emplace_back(slack, i);
        }
        std::sort(order.begin(), order.end());

        for (const auto& item : order) {
            const int i = item.second;
            double vx = mec.c.x - s[i].x;
            double vy = mec.c.y - s[i].y;
            double vn = std::hypot(vx, vy);
            if (vn > 1e-10) {
                vx /= vn;
                vy /= vn;
                const double boundary_bias = std::max(0.0, 0.012 - item.first);
                const double step = strength * clampd(0.0012 + 0.55 * boundary_bias, 0.0004, 0.018);
                const double tangent = gauss(rng, 0.0008 + 0.0015 * strength);
                s[i].x += step * vx - tangent * vy;
                s[i].y += step * vy + tangent * vx;
            }

            const double target_theta = best_theta_for_center(s[i], mec.c);
            s[i].theta = wrap_theta(s[i].theta + 0.38 * angle_distance(target_theta, s[i].theta));
        }

        quick_repair(s, rng, 1 + (strength > 0.50 ? 1 : 0), 40, 40);
        recenter_mean(s);
    }
}

void micro_polish(std::vector<Semicircle>& s,
                  const EvalConfig& cfg,
                  double penalty_weight,
                  std::mt19937_64& rng,
                  int passes,
                  double pos_step,
                  double theta_step) {
    FastResult best = fast_eval(s, cfg, penalty_weight);
    for (int pass = 0; pass < passes; ++pass) {
        const Circle mec = compute_mec(s, std::max(16, cfg.mec_boundary_points));
        std::vector<std::pair<double, int>> order;
        order.reserve(N);
        for (int i = 0; i < N; ++i) {
            const Point f = farthest_boundary_point_from(s[i], mec.c.x, mec.c.y);
            order.emplace_back(mec.r - dist(f, mec.c), i);
        }
        std::sort(order.begin(), order.end());

        bool improved = false;
        for (const auto& item : order) {
            const int i = item.second;
            const Semicircle orig = s[i];
            bool accepted = false;

            double vx = mec.c.x - orig.x;
            double vy = mec.c.y - orig.y;
            double vn = std::hypot(vx, vy);
            if (vn < 1e-10) {
                const double ang = TAU * rand01(rng);
                vx = std::cos(ang);
                vy = std::sin(ang);
                vn = 1.0;
            }
            vx /= vn;
            vy /= vn;
            const double tx = -vy;
            const double ty = vx;
            const double target_theta = best_theta_for_center(orig, mec.c);

            const std::array<Semicircle, 7> probes{
                Semicircle{orig.x + pos_step * vx, orig.y + pos_step * vy, orig.theta},
                Semicircle{orig.x - 0.45 * pos_step * vx, orig.y - 0.45 * pos_step * vy, orig.theta},
                Semicircle{orig.x + 0.70 * pos_step * tx, orig.y + 0.70 * pos_step * ty, orig.theta},
                Semicircle{orig.x - 0.70 * pos_step * tx, orig.y - 0.70 * pos_step * ty, orig.theta},
                Semicircle{orig.x, orig.y, wrap_theta(orig.theta + theta_step)},
                Semicircle{orig.x, orig.y, wrap_theta(orig.theta - theta_step)},
                Semicircle{orig.x + 0.50 * pos_step * vx, orig.y + 0.50 * pos_step * vy, target_theta}
            };

            for (const Semicircle& probe : probes) {
                std::vector<Semicircle> cand = s;
                cand[i] = probe;
                recenter_mean(cand);
                const FastResult cand_eval = fast_eval(cand, cfg, penalty_weight);
                if (cand_eval.energy + 1e-12 < best.energy) {
                    s = std::move(cand);
                    best = cand_eval;
                    improved = true;
                    accepted = true;
                    break;
                }
            }
            if (!accepted) continue;
        }

        if (!improved) {
            pos_step *= 0.5;
            theta_step *= 0.6;
            if (pos_step < 1e-4 && theta_step < 5e-4) break;
        }
    }
}

bool exact_nudge_pass(std::vector<Semicircle>& s,
                      const SearchParams& p,
                      std::mt19937_64& rng,
                      ExactResult& best_exact) {
    if (!best_exact.valid) return false;

    std::vector<std::pair<double, int>> order;
    order.reserve(N);
    for (int i = 0; i < N; ++i) {
        const Point f = farthest_boundary_point_from(s[i], best_exact.mec_center.x, best_exact.mec_center.y);
        order.emplace_back(best_exact.radius - dist(f, best_exact.mec_center), i);
    }
    std::sort(order.begin(), order.end());

    const std::array<double, 3> pos_steps{0.0012, 0.00045, 0.00018};
    const std::array<double, 3> theta_steps{0.010, 0.004, 0.0016};
    const int limit = std::min(N, 8);

    for (int si = 0; si < static_cast<int>(pos_steps.size()); ++si) {
        const double pos_step = pos_steps[si];
        const double theta_step = theta_steps[si];

        for (int ord = 0; ord < limit; ++ord) {
            const int i = order[ord].second;
            const Semicircle orig = s[i];

            double vx = best_exact.mec_center.x - orig.x;
            double vy = best_exact.mec_center.y - orig.y;
            double vn = std::hypot(vx, vy);
            if (vn < 1e-10) {
                const double ang = TAU * rand01(rng);
                vx = std::cos(ang);
                vy = std::sin(ang);
                vn = 1.0;
            }
            vx /= vn;
            vy /= vn;
            const double tx = -vy;
            const double ty = vx;
            const double target_theta = best_theta_for_center(orig, best_exact.mec_center);

            const std::array<Semicircle, 8> probes{
                Semicircle{orig.x + pos_step * vx, orig.y + pos_step * vy, orig.theta},
                Semicircle{orig.x - 0.35 * pos_step * vx, orig.y - 0.35 * pos_step * vy, orig.theta},
                Semicircle{orig.x + 0.65 * pos_step * tx, orig.y + 0.65 * pos_step * ty, orig.theta},
                Semicircle{orig.x - 0.65 * pos_step * tx, orig.y - 0.65 * pos_step * ty, orig.theta},
                Semicircle{orig.x, orig.y, wrap_theta(orig.theta + theta_step)},
                Semicircle{orig.x, orig.y, wrap_theta(orig.theta - theta_step)},
                Semicircle{orig.x + 0.45 * pos_step * vx, orig.y + 0.45 * pos_step * vy, target_theta},
                Semicircle{orig.x + gauss(rng, 0.35 * pos_step), orig.y + gauss(rng, 0.35 * pos_step),
                           wrap_theta(target_theta + gauss(rng, 0.35 * theta_step))}
            };

            for (const Semicircle& probe : probes) {
                std::vector<Semicircle> cand = s;
                cand[i] = probe;
                recenter_mean(cand);

                const FastResult fast = fast_eval(cand, p.fast_cfg, p.penalty_weight);
                if (fast.overlap_sum > 3e-6 || fast.containment_excess > 1e-9) continue;
                if (fast.radius > best_exact.radius + 0.012) continue;

                const ExactResult ex = exact_validate(cand, p.exact_cfg);
                if (ex.valid && ex.radius + 1e-12 < best_exact.radius) {
                    s = round_solution(cand);
                    best_exact = ex;
                    return true;
                }
            }
        }
    }
    return false;
}

bool exact_pair_nudge_pass(std::vector<Semicircle>& s,
                           const SearchParams& p,
                           std::mt19937_64& rng,
                           ExactResult& best_exact) {
    if (!best_exact.valid) return false;

    struct PairItem {
        double key;
        int i;
        int j;
    };

    std::vector<PairItem> pairs;
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            const double d = std::hypot(s[i].x - s[j].x, s[i].y - s[j].y);
            if (d < 2.25) pairs.push_back(PairItem{std::abs(d - 2.0), i, j});
        }
    }
    std::sort(pairs.begin(), pairs.end(), [](const PairItem& a, const PairItem& b) {
        return a.key < b.key;
    });
    if (pairs.empty()) return false;

    const std::array<double, 2> pos_steps{0.0008, 0.0003};
    const std::array<double, 2> theta_steps{0.006, 0.0022};
    const int limit = std::min(static_cast<int>(pairs.size()), 12);

    for (int si = 0; si < static_cast<int>(pos_steps.size()); ++si) {
        const double pos_step = pos_steps[si];
        const double theta_step = theta_steps[si];

        for (int ord = 0; ord < limit; ++ord) {
            const int i = pairs[ord].i;
            const int j = pairs[ord].j;
            const Semicircle a0 = s[i];
            const Semicircle b0 = s[j];

            double aix = best_exact.mec_center.x - a0.x;
            double aiy = best_exact.mec_center.y - a0.y;
            double ajx = best_exact.mec_center.x - b0.x;
            double ajy = best_exact.mec_center.y - b0.y;
            double ain = std::hypot(aix, aiy);
            double ajn = std::hypot(ajx, ajy);
            if (ain < 1e-10) {
                const double ang = TAU * rand01(rng);
                aix = std::cos(ang);
                aiy = std::sin(ang);
                ain = 1.0;
            }
            if (ajn < 1e-10) {
                const double ang = TAU * rand01(rng);
                ajx = std::cos(ang);
                ajy = std::sin(ang);
                ajn = 1.0;
            }
            aix /= ain; aiy /= ain;
            ajx /= ajn; ajy /= ajn;

            double px = a0.x - b0.x;
            double py = a0.y - b0.y;
            double pn = std::hypot(px, py);
            if (pn < 1e-10) {
                px = -aiy;
                py = aix;
                pn = 1.0;
            }
            px /= pn; py /= pn;
            const double qx = -py;
            const double qy = px;

            const double ta = best_theta_for_center(a0, best_exact.mec_center);
            const double tb = best_theta_for_center(b0, best_exact.mec_center);

            struct PairProbe {
                Semicircle a;
                Semicircle b;
            };
            const std::array<PairProbe, 7> probes{
                PairProbe{Semicircle{a0.x + pos_step * aix, a0.y + pos_step * aiy, a0.theta},
                          Semicircle{b0.x + pos_step * ajx, b0.y + pos_step * ajy, b0.theta}},
                PairProbe{Semicircle{a0.x + 0.45 * pos_step * aix, a0.y + 0.45 * pos_step * aiy, ta},
                          Semicircle{b0.x + 0.45 * pos_step * ajx, b0.y + 0.45 * pos_step * ajy, tb}},
                PairProbe{Semicircle{a0.x + 0.75 * pos_step * qx, a0.y + 0.75 * pos_step * qy, a0.theta},
                          Semicircle{b0.x + 0.75 * pos_step * qx, b0.y + 0.75 * pos_step * qy, b0.theta}},
                PairProbe{Semicircle{a0.x + 0.65 * pos_step * px, a0.y + 0.65 * pos_step * py, a0.theta},
                          Semicircle{b0.x - 0.65 * pos_step * px, b0.y - 0.65 * pos_step * py, b0.theta}},
                PairProbe{Semicircle{a0.x, a0.y, wrap_theta(a0.theta + theta_step)},
                          Semicircle{b0.x, b0.y, wrap_theta(b0.theta - theta_step)}},
                PairProbe{Semicircle{a0.x + 0.30 * pos_step * aix, a0.y + 0.30 * pos_step * aiy, ta},
                          Semicircle{b0.x + 0.30 * pos_step * ajx, b0.y + 0.30 * pos_step * ajy, wrap_theta(tb + theta_step)}},
                PairProbe{Semicircle{a0.x + gauss(rng, 0.25 * pos_step), a0.y + gauss(rng, 0.25 * pos_step),
                                     wrap_theta(ta + gauss(rng, 0.25 * theta_step))},
                          Semicircle{b0.x + gauss(rng, 0.25 * pos_step), b0.y + gauss(rng, 0.25 * pos_step),
                                     wrap_theta(tb + gauss(rng, 0.25 * theta_step))}}
            };

            for (const PairProbe& probe : probes) {
                std::vector<Semicircle> cand = s;
                cand[i] = probe.a;
                cand[j] = probe.b;
                recenter_mean(cand);

                const FastResult fast = fast_eval(cand, p.fast_cfg, p.penalty_weight);
                if (fast.overlap_sum > 3e-6 || fast.containment_excess > 1e-9) continue;
                if (fast.radius > best_exact.radius + 0.010) continue;

                const ExactResult ex = exact_validate(cand, p.exact_cfg);
                if (ex.valid && ex.radius + 1e-12 < best_exact.radius) {
                    s = round_solution(cand);
                    best_exact = ex;
                    return true;
                }
            }
        }
    }

    return false;
}

std::vector<Semicircle> mutate(const std::vector<Semicircle>& cur,
                               double progress,
                               std::mt19937_64& rng,
                               const std::vector<double>& weights,
                               const std::vector<std::pair<int, int>>& pair_list,
                               double scale_mul) {
    std::vector<Semicircle> cand = cur;

    const double scale = std::max(0.15, scale_mul);
    const double pos_small = scale * (0.015 * (1.0 - progress) + 0.0008);
    const double pos_large = scale * (0.055 * (1.0 - progress) + 0.0035);
    const double th_small = scale * (0.030 * (1.0 - progress) + 0.0012);
    const double r = rand01(rng);

    if (r < 0.30) {
        const int i = weighted_pick(rng, weights);
        cand[i].x += gauss(rng, pos_small);
        cand[i].y += gauss(rng, pos_small);
        cand[i].theta += gauss(rng, th_small);
    } else if (r < 0.53) {
        const auto [i, j] = pair_list[std::uniform_int_distribution<int>(0, static_cast<int>(pair_list.size()) - 1)(rng)];
        const double dx = gauss(rng, pos_small);
        const double dy = gauss(rng, pos_small);
        const double dt = gauss(rng, th_small);
        cand[i].x += dx;
        cand[i].y += dy;
        cand[i].theta += dt;
        cand[j].x += dx;
        cand[j].y += dy;
        cand[j].theta -= dt;
    } else if (r < 0.73) {
        const auto [i, j] = pair_list[std::uniform_int_distribution<int>(0, static_cast<int>(pair_list.size()) - 1)(rng)];
        double vx = cand[i].x - cand[j].x;
        double vy = cand[i].y - cand[j].y;
        double vn = std::hypot(vx, vy);
        if (vn < 1e-9) {
            const double a = TAU * rand01(rng);
            vx = std::cos(a);
            vy = std::sin(a);
            vn = 1.0;
        }
        vx /= vn;
        vy /= vn;
        const double step = std::abs(gauss(rng, pos_large)) * 0.45;
        cand[i].x += vx * step;
        cand[i].y += vy * step;
        cand[j].x -= vx * step;
        cand[j].y -= vy * step;
        cand[i].theta += gauss(rng, th_small);
        cand[j].theta += gauss(rng, th_small);
    } else if (r < 0.88) {
        const int k = (rand01(rng) < 0.7 ? 2 : 3);
        const double dx = gauss(rng, pos_small);
        const double dy = gauss(rng, pos_small);
        std::vector<int> idxs;
        idxs.reserve(static_cast<size_t>(k));
        while (static_cast<int>(idxs.size()) < k) {
            const int i = weighted_pick(rng, weights);
            if (std::find(idxs.begin(), idxs.end(), i) == idxs.end()) idxs.push_back(i);
        }
        for (int i : idxs) {
            cand[i].x += dx + gauss(rng, pos_small * 0.45);
            cand[i].y += dy + gauss(rng, pos_small * 0.45);
            cand[i].theta += gauss(rng, th_small);
        }
    } else {
        const double rot = gauss(rng, 0.04 * (1.0 - progress) + 0.002);
        const double scale = 1.0 + gauss(rng, 0.005 * (1.0 - progress) + 0.0004);
        const double cc = std::cos(rot), ss = std::sin(rot);
        for (Semicircle& sc : cand) {
            const double x = sc.x;
            const double y = sc.y;
            sc.x = scale * (cc * x - ss * y);
            sc.y = scale * (ss * x + cc * y);
            sc.theta += rot + gauss(rng, th_small * 0.4);
        }
    }

    if (rand01(rng) < 0.30) recenter_mean(cand);
    for (Semicircle& sc : cand) sc.theta = wrap_theta(sc.theta);
    return cand;
}

void quick_repair(std::vector<Semicircle>& s,
                  std::mt19937_64& rng,
                  int iters = 3,
                  int overlap_arc = 48,
                  int mec_pts = 48) {
    for (int it = 0; it < iters; ++it) {
        for (int i = 0; i < N; ++i) {
            for (int j = i + 1; j < N; ++j) {
                const double dx = s[i].x - s[j].x;
                const double dy = s[i].y - s[j].y;
                if (dx * dx + dy * dy > 4.84) continue;
                const double a = overlap_area_polygonized(s[i], s[j], overlap_arc);
                if (a <= 5e-5) continue;

                double vx = dx;
                double vy = dy;
                double vn = std::hypot(vx, vy);
                if (vn < 1e-9) {
                    const double ang = TAU * rand01(rng);
                    vx = std::cos(ang);
                    vy = std::sin(ang);
                    vn = 1.0;
                }
                vx /= vn;
                vy /= vn;

                const double step = clampd(0.002 + 0.040 * std::sqrt(a), 0.001, 0.025);
                s[i].x += step * vx;
                s[i].y += step * vy;
                s[j].x -= step * vx;
                s[j].y -= step * vy;
                s[i].theta += gauss(rng, 0.004);
                s[j].theta += gauss(rng, 0.004);
            }
        }

        const Circle mec = compute_mec(s, mec_pts);
        for (int i = 0; i < N; ++i) {
            const Point f = farthest_boundary_point_from(s[i], mec.c.x, mec.c.y);
            const double ex = dist(f, mec.c) - mec.r;
            if (ex <= 0.0) continue;

            double vx = s[i].x - mec.c.x;
            double vy = s[i].y - mec.c.y;
            double vn = std::hypot(vx, vy);
            if (vn < 1e-9) {
                vx = f.x - mec.c.x;
                vy = f.y - mec.c.y;
                vn = std::hypot(vx, vy);
                if (vn < 1e-9) {
                    const double ang = TAU * rand01(rng);
                    vx = std::cos(ang);
                    vy = std::sin(ang);
                    vn = 1.0;
                }
            }
            vx /= vn;
            vy /= vn;

            const double step = clampd(0.9 * ex, 0.0, 0.03);
            s[i].x -= step * vx;
            s[i].y -= step * vy;
        }

        if (it + 1 != iters) recenter_mean(s);
    }
}

void maybe_update_global(GlobalBest& global,
                         const std::vector<Semicircle>& cand,
                         double score,
                         const std::string& out_path,
                         Logger& logger) {
    bool improved = false;
    std::string line;
    {
        std::lock_guard<std::mutex> lock(global.mu);
        if (score + 1e-12 < global.score) {
            global.score = score;
            global.sol = round_solution(cand);
            global.improvements++;
            global.last_improve_tp = std::chrono::steady_clock::now();
            save_solution_json(out_path, global.sol);

            std::ostringstream oss;
            oss << "[" << now_str() << "] improved exact score: "
                << std::fixed << std::setprecision(12) << score
                << "  improvements=" << global.improvements;
            line = oss.str();
            improved = true;
        }
    }
    if (improved) log_message(logger, line);
}

void progress_reporter_loop(const SearchParams& p,
                            GlobalBest& global,
                            std::vector<WorkerSnapshot>& worker_stats,
                            const std::chrono::steady_clock::time_point start_tp,
                            const std::chrono::steady_clock::time_point deadline,
                            std::atomic<bool>& stop_flag,
                            Logger& logger) {
    const int interval = std::max(1, p.report_every);
    auto next_tick = start_tp + std::chrono::seconds(interval);

    while (!stop_flag.load(std::memory_order_relaxed)) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= next_tick) {
            double best_score = HUGE_SCORE;
            uint64_t improvements = 0;
            double since_improve = 0.0;
            {
                std::lock_guard<std::mutex> lock(global.mu);
                best_score = global.score;
                improvements = global.improvements;
                if (improvements > 0) {
                    since_improve = std::chrono::duration<double>(now - global.last_improve_tp).count();
                }
            }
            const double elapsed = std::chrono::duration<double>(now - start_tp).count();
            const double remaining = std::max(0.0, std::chrono::duration<double>(deadline - now).count());

            std::vector<double> radii;
            std::vector<double> overlaps;
            std::vector<double> containments;
            std::vector<double> local_best_radii;
            uint64_t total_iters = 0;
            uint64_t total_accepts = 0;
            uint64_t total_exact_checks = 0;
            uint64_t total_exact_valid = 0;
            int near_frontier = 0;
            int approx_feasible = 0;

            for (auto& ws : worker_stats) {
                std::lock_guard<std::mutex> wlock(ws.mu);
                if (!ws.has) continue;
                radii.push_back(ws.cur_radius);
                overlaps.push_back(ws.cur_overlap);
                containments.push_back(ws.cur_containment);
                local_best_radii.push_back(ws.local_best_radius);
                total_iters += ws.iter;
                total_accepts += ws.accepts;
                total_exact_checks += ws.exact_checks;
                total_exact_valid += ws.exact_valid;
                if (ws.cur_radius < best_score + 0.010) near_frontier++;
                if (ws.cur_overlap <= p.fast_cfg.overlap_tol && ws.cur_containment <= 1e-10) approx_feasible++;
            }

            auto minv = [](const std::vector<double>& v) {
                return v.empty() ? HUGE_SCORE : *std::min_element(v.begin(), v.end());
            };
            auto maxv = [](const std::vector<double>& v) {
                return v.empty() ? HUGE_SCORE : *std::max_element(v.begin(), v.end());
            };

            std::ostringstream oss;
            oss << "[" << now_str() << "] progress elapsed=" << std::fixed << std::setprecision(1) << elapsed
                << "s remaining=" << remaining << "s"
                << " best=" << std::setprecision(12) << best_score
                << " improvements=" << improvements
                << " | scan_r(min/med/max)=" << std::setprecision(6)
                << minv(radii) << "/" << median_of(radii) << "/" << maxv(radii)
                << " ov(min/med/max)="
                << minv(overlaps) << "/" << median_of(overlaps) << "/" << maxv(overlaps)
                << " cx(min/med/max)="
                << minv(containments) << "/" << median_of(containments) << "/" << maxv(containments)
                << " local_best_min=" << minv(local_best_radii)
                << " near_frontier=" << near_frontier
                << " approx_feasible=" << approx_feasible
                << " exact_checks=" << total_exact_checks
                << " exact_valid=" << total_exact_valid
                << " accepts=" << total_accepts
                << " total_iters=" << total_iters;
            if (improvements > 0) {
                oss << " since_last_improve=" << std::setprecision(1) << since_improve << "s";
            }
            log_message(logger, oss.str());
            next_tick += std::chrono::seconds(interval);
            continue;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}

void worker_loop(int tid,
                 const SearchParams& p,
                 const std::vector<Semicircle>& base,
                 const std::vector<double>& weights,
                 const std::vector<std::pair<int, int>>& pair_list,
                 GlobalBest& global,
                 WorkerSnapshot& snapshot,
                 Logger& logger,
                 std::atomic<bool>& stop_flag,
                 const std::chrono::steady_clock::time_point deadline) {
    std::mt19937_64 rng((p.seed + 0x9E3779B97F4A7C15ULL * static_cast<uint64_t>(tid + 1)) ^
                        (0xBF58476D1CE4E5B9ULL * static_cast<uint64_t>(tid + 11)));
    const WorkerProfile profile = make_worker_profile(tid);

    std::vector<Semicircle> local = base;
    for (int i = 0; i < N; ++i) {
        local[i].x += gauss(rng, profile.init_jitter);
        local[i].y += gauss(rng, profile.init_jitter);
        local[i].theta += gauss(rng, std::max(0.0005, 0.6 * profile.init_jitter + 0.0015));
    }
    recenter_mean(local);
    if (profile.start_from_best) {
        micro_polish(local, p.fast_cfg, p.penalty_weight, rng, 1, 0.0009, 0.006);
    }

    FastResult cur = fast_eval(local, p.fast_cfg, p.penalty_weight);
    std::vector<Semicircle> local_best = local;
    FastResult local_best_eval = cur;
    ExactResult local_exact;
    if (profile.start_from_best && cur.approx_valid) {
        local_exact = exact_validate(local, p.exact_cfg);
    }

    uint64_t accepted_moves = 0;
    uint64_t exact_checks = 0;
    uint64_t exact_valid = 0;
    uint64_t last_iter = 0;
    int stagnation = 0;
    double announced_approx_radius = cur.radius;
    double announced_exact_radius = HUGE_SCORE;

    for (size_t iter = 1;
         !stop_flag.load(std::memory_order_relaxed) && std::chrono::steady_clock::now() < deadline;
         ++iter) {
        last_iter = static_cast<uint64_t>(iter);
        const double cycle = static_cast<double>(iter % 5000) / 5000.0;

        std::vector<Semicircle> cand = mutate(local, cycle, rng, weights, pair_list, profile.mutation_scale);
        if (!profile.start_from_best && rand01(rng) < profile.compaction_rate) {
            const double strength = profile.compaction_strength * (0.35 + 0.30 * rand01(rng));
            pressure_compact(cand, rng, strength, 1 + (rand01(rng) < 0.35 ? 1 : 0));
        }
        if (rand01(rng) < profile.repair_rate) {
            quick_repair(cand, rng, profile.repair_iters, 40, 40);
        }
        if (profile.start_from_best && rand01(rng) < 0.10) {
            micro_polish(cand, p.fast_cfg, p.penalty_weight, rng, 1, 0.0007, 0.0045);
        }

        FastResult cand_eval = fast_eval(cand, p.fast_cfg, p.penalty_weight);

        const double T0 = 0.015;
        const double T1 = 0.00035;
        const double T = T0 * std::pow(T1 / T0, cycle);
        const double dE = cand_eval.energy - cur.energy;

        bool accept = false;
        if (dE <= 0.0) accept = true;
        else if (rand01(rng) < std::exp(-dE / std::max(1e-12, T))) accept = true;

        if (accept) {
            local = std::move(cand);
            cur = cand_eval;
            accepted_moves++;
            if (cur.approx_valid && cur.radius + 2e-4 < announced_approx_radius) {
                announced_approx_radius = cur.radius;
                std::ostringstream oss;
                oss << "[" << now_str() << "] worker=" << tid
                    << " iter=" << iter
                    << " approx_candidate radius=" << std::fixed << std::setprecision(12) << cur.radius
                    << " overlap=" << std::setprecision(8) << cur.overlap_sum
                    << " contain_excess=" << cur.containment_excess
                    << " energy=" << cur.energy;
                log_message(logger, oss.str());
            }
            if (cur.energy + 1e-15 < local_best_eval.energy) {
                local_best = local;
                local_best_eval = cur;
                stagnation = 0;
            } else {
                ++stagnation;
            }
        } else {
            ++stagnation;
        }

        double global_score_snapshot;
        std::vector<Semicircle> global_sol_snapshot;
        {
            std::lock_guard<std::mutex> lock(global.mu);
            global_score_snapshot = global.score;
            global_sol_snapshot = global.sol;
        }

        if (cur.overlap_sum < 4e-4 &&
            cur.containment_excess < 2e-6 &&
            cur.radius < global_score_snapshot + profile.exact_margin) {
            exact_checks++;
            ExactResult ex = exact_validate(local, p.exact_cfg);
            if (ex.valid) {
                exact_valid++;
                if (ex.radius + 5e-6 < announced_exact_radius) {
                    announced_exact_radius = ex.radius;
                    std::ostringstream oss;
                    oss << "[" << now_str() << "] worker=" << tid
                        << " iter=" << iter
                        << " exact_candidate radius=" << std::fixed << std::setprecision(12) << ex.radius
                        << " overlap_sum=" << std::setprecision(10) << ex.overlap_sum
                        << " mec=(" << ex.mec_center.x << "," << ex.mec_center.y << ")";
                    log_message(logger, oss.str());
                }
                if (ex.radius + 1e-12 < global_score_snapshot) {
                    maybe_update_global(global, local, ex.radius, p.output_path, logger);
                }
            }
        }

        if (iter % 2500 == 0) {
            if (profile.start_from_best) {
                std::vector<Semicircle> exact_seed = global_sol_snapshot.empty() ? local : global_sol_snapshot;
                ExactResult probe = exact_validate(exact_seed, p.exact_cfg);
                bool improved_exact = probe.valid && exact_nudge_pass(exact_seed, p, rng, probe);
                if (!improved_exact && probe.valid) {
                    improved_exact = exact_pair_nudge_pass(exact_seed, p, rng, probe);
                }
                if (improved_exact) {
                    local = exact_seed;
                    cur = fast_eval(local, p.fast_cfg, p.penalty_weight);
                    local_best = local;
                    local_best_eval = cur;
                    local_exact = probe;
                    maybe_update_global(global, local, probe.radius, p.output_path, logger);
                    stagnation = 0;
                    continue;
                }
            }
            if (profile.start_from_best && !global_sol_snapshot.empty() && rand01(rng) < 0.70) {
                local = global_sol_snapshot;
                micro_polish(local, p.fast_cfg, p.penalty_weight, rng, 2, 0.0012, 0.0075);
                for (int k = 0; k < N; ++k) {
                    local[k].x += gauss(rng, 0.0002 + 0.0005 * profile.mutation_scale);
                    local[k].y += gauss(rng, 0.0002 + 0.0005 * profile.mutation_scale);
                    local[k].theta += gauss(rng, 0.0004 + 0.0010 * profile.mutation_scale);
                }
                recenter_mean(local);
                cur = fast_eval(local, p.fast_cfg, p.penalty_weight);
                local_best = local;
                local_best_eval = cur;
                stagnation = 0;
            } else if (local_best_eval.energy + 1e-12 < cur.energy) {
                local = local_best;
                cur = local_best_eval;
                stagnation = 0;
            } else if (!global_sol_snapshot.empty() && rand01(rng) < 0.60) {
                local = global_sol_snapshot;
                if (!profile.start_from_best && rand01(rng) < 0.45) {
                    pressure_compact(local, rng, 0.10 + 0.45 * profile.compaction_strength, 1);
                }
                for (int k = 0; k < N; ++k) {
                    local[k].x += gauss(rng, 0.0006 + 0.0012 * profile.mutation_scale);
                    local[k].y += gauss(rng, 0.0006 + 0.0012 * profile.mutation_scale);
                    local[k].theta += gauss(rng, 0.0012 + 0.0020 * profile.mutation_scale);
                }
                recenter_mean(local);
                cur = fast_eval(local, p.fast_cfg, p.penalty_weight);
                local_best = local;
                local_best_eval = cur;
                stagnation = 0;
            }
        }

        if (stagnation > (profile.start_from_best ? 4200 : 6000)) {
            if (!global_sol_snapshot.empty()) {
                local = global_sol_snapshot;
                if (profile.start_from_best) {
                    micro_polish(local, p.fast_cfg, p.penalty_weight, rng, 3, 0.0015, 0.0100);
                    ExactResult probe = exact_validate(local, p.exact_cfg);
                    bool improved_exact = probe.valid && exact_nudge_pass(local, p, rng, probe);
                    if (!improved_exact && probe.valid) {
                        improved_exact = exact_pair_nudge_pass(local, p, rng, probe);
                    }
                    if (improved_exact) {
                        local_exact = probe;
                        maybe_update_global(global, local, probe.radius, p.output_path, logger);
                    }
                } else if (rand01(rng) < 0.60) {
                    pressure_compact(local, rng, 0.12 + 0.55 * profile.compaction_strength, 1 + (rand01(rng) < 0.25 ? 1 : 0));
                }
                for (int k = 0; k < N; ++k) {
                    local[k].x += gauss(rng, 0.0005 + 0.0018 * profile.mutation_scale);
                    local[k].y += gauss(rng, 0.0005 + 0.0018 * profile.mutation_scale);
                    local[k].theta += gauss(rng, 0.0010 + 0.0035 * profile.mutation_scale);
                }
            } else {
                local = base;
                if (rand01(rng) < 0.50) {
                    pressure_compact(local, rng, 0.10 + 0.45 * profile.compaction_strength, 1);
                }
                for (int k = 0; k < N; ++k) {
                    local[k].x += gauss(rng, 0.0015 + 0.0035 * profile.mutation_scale);
                    local[k].y += gauss(rng, 0.0015 + 0.0035 * profile.mutation_scale);
                    local[k].theta += gauss(rng, 0.0030 + 0.0080 * profile.mutation_scale);
                }
            }
            recenter_mean(local);
            if (rand01(rng) < profile.repair_rate) {
                quick_repair(local, rng, profile.repair_iters + 1, 40, 40);
            }
            cur = fast_eval(local, p.fast_cfg, p.penalty_weight);
            local_best = local;
            local_best_eval = cur;
            if (profile.start_from_best && cur.approx_valid) {
                local_exact = exact_validate(local, p.exact_cfg);
            }
            stagnation = 0;
        }

        if ((iter & 255ULL) == 0ULL) {
            publish_worker_snapshot(snapshot,
                                    static_cast<uint64_t>(iter),
                                    accepted_moves,
                                    exact_checks,
                                    exact_valid,
                                    cur,
                                    local_best_eval,
                                    stagnation);
        }
    }

    publish_worker_snapshot(snapshot,
                            last_iter,
                            accepted_moves,
                            exact_checks,
                            exact_valid,
                            cur,
                            local_best_eval,
                            stagnation);
}

SearchParams parse_args(int argc, char** argv) {
    SearchParams p;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&](const std::string& name) {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + name);
            return std::string(argv[++i]);
        };

        if (a == "--input") p.input_path = need(a);
        else if (a == "--output") p.output_path = need(a);
        else if (a == "--log-file") p.log_path = need(a);
        else if (a == "--threads") p.threads = std::max(1, std::stoi(need(a)));
        else if (a == "--seconds") p.seconds = std::max(1, std::stoi(need(a)));
        else if (a == "--report-every" || a == "--log-every") p.report_every = std::max(0, std::stoi(need(a)));
        else if (a == "--seed") p.seed = std::stoull(need(a));
        else if (a == "--search-arc") p.fast_cfg.arc_points = std::max(8, std::stoi(need(a)));
        else if (a == "--search-mec") p.fast_cfg.mec_boundary_points = std::max(8, std::stoi(need(a)));
        else if (a == "--search-overlap-tol") p.fast_cfg.overlap_tol = std::max(0.0, std::stod(need(a)));
        else if (a == "--exact-arc") p.exact_cfg.arc_points = std::max(128, std::stoi(need(a)));
        else if (a == "--exact-mec") p.exact_cfg.mec_boundary_points = std::max(16, std::stoi(need(a)));
        else if (a == "--penalty") p.penalty_weight = std::stod(need(a));
        else if (a == "--score-only") p.score_only = true;
        else if (a == "-h" || a == "--help") {
            std::cout << "Usage: ./packing [--input best_solution.json] [--output best_solution.json] [--threads 16] [--seconds 300]\n"
                      << "                [--report-every 0] [--log-file best_solution.log] [--seed 1] [--penalty 260]\n"
                      << "                [--search-arc 64] [--search-mec 32] [--search-overlap-tol 5e-6]\n"
                      << "                [--exact-arc 4096] [--exact-mec 128] [--score-only]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + a);
        }
    }
    return p;
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::ios::sync_with_stdio(false);
        std::cin.tie(nullptr);

        const SearchParams p = parse_args(argc, argv);
        Logger logger;
        logger.file.open(p.log_path, std::ios::out | std::ios::trunc);
        if (!logger.file) throw std::runtime_error("Failed to open log file: " + p.log_path);

        std::vector<Semicircle> base;
        if (!load_solution_json(p.input_path, base)) {
            throw std::runtime_error("Failed to load input JSON: " + p.input_path);
        }

        if (static_cast<int>(base.size()) != N) {
            throw std::runtime_error("Input must contain exactly 15 semicircles");
        }

        ExactResult base_exact = exact_validate(base, p.exact_cfg);
        if (!base_exact.valid) {
            log_message(logger, "Input INVALID: " + base_exact.error);
            return 1;
        }

        {
            std::ostringstream oss;
            oss << "[" << now_str() << "] run start"
                << " input=" << p.input_path
                << " output=" << p.output_path
                << " log=" << p.log_path
                << " threads=" << p.threads
                << " seconds=" << p.seconds
                << " report_every=" << p.report_every
                << " seed=" << p.seed;
            log_message(logger, oss.str());
        }
        {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(12)
                << "Base exact score: " << base_exact.radius
                << "  MEC=(" << base_exact.mec_center.x << ", " << base_exact.mec_center.y << ")";
            log_message(logger, oss.str());
        }

        GlobalBest global;
        global.score = base_exact.radius;
        global.sol = round_solution(base);
        global.improvements = 1;
        global.last_improve_tp = std::chrono::steady_clock::now();
        save_solution_json(p.output_path, global.sol);
        log_message(logger, "Initial best written to " + p.output_path);

        if (p.score_only) return 0;

        const std::vector<double> weights = build_weights(global.sol);
        const std::vector<std::pair<int, int>> pair_list = build_pair_list(global.sol);

        const auto start_tp = std::chrono::steady_clock::now();
        std::atomic<bool> stop_flag{false};
        const auto deadline = start_tp + std::chrono::seconds(p.seconds);
        std::vector<WorkerSnapshot> worker_stats(static_cast<size_t>(p.threads));

        std::thread reporter;
        if (p.report_every > 0) {
            reporter = std::thread(progress_reporter_loop,
                                   std::cref(p),
                                   std::ref(global),
                                   std::ref(worker_stats),
                                   start_tp,
                                   deadline,
                                   std::ref(stop_flag),
                                   std::ref(logger));
        }
        std::vector<std::thread> threads;
        threads.reserve(static_cast<size_t>(p.threads));
        for (int t = 0; t < p.threads; ++t) {
            threads.emplace_back(worker_loop,
                                 t,
                                 std::cref(p),
                                 std::cref(base),
                                 std::cref(weights),
                                 std::cref(pair_list),
                                 std::ref(global),
                                 std::ref(worker_stats[static_cast<size_t>(t)]),
                                 std::ref(logger),
                                 std::ref(stop_flag),
                                 deadline);
        }

        for (auto& th : threads) th.join();
        stop_flag.store(true, std::memory_order_relaxed);
        if (reporter.joinable()) reporter.join();

        ExactResult final_exact = exact_validate(global.sol, p.exact_cfg);
        if (!final_exact.valid) {
            log_message(logger, "Final best failed exact validation: " + final_exact.error);
            return 2;
        }

        save_solution_json(p.output_path, global.sol);
        {
            std::ostringstream oss;
            oss << "Final exact score: " << std::fixed << std::setprecision(12) << final_exact.radius;
            log_message(logger, oss.str());
        }
        log_message(logger, "Saved: " + p.output_path);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
