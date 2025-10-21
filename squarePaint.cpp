#include <bits/stdc++.h>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <iomanip>
using namespace std;
using boost::multiprecision::cpp_int;
using boost::multiprecision::cpp_dec_float_100;
static void print_asymptotic_approx(int N, int a) {
    long double n = (long double)N;
    long double d = (long double)a;
    long double L = lgammal(d*n + 1.0L) - 2.0L * n * lgammal(d + 1.0L) - 0.5L * (d - 1.0L) * (d - 1.0L);
    long double log10v = L / logl(10.0L);
    long long exp10 = (long long)floorl(log10v);
    long double frac = log10v - (long double)exp10;
    long double mant = powl(10.0L, frac);
    std::ostringstream oss;
    oss.setf(std::ios::fmtflags(0), std::ios::floatfield);
    oss << std::setprecision(20) << mant << "e+" << exp10;
    cout << oss.str() << "\n";
}

static inline unsigned hw_threads() {
    unsigned t = thread::hardware_concurrency();
    return t ? t : 1u;
}

struct StrHash {
    size_t operator()(const string& s) const noexcept {
        static const uint64_t FIXED = 0x9e3779b97f4a7c15ull;
        uint64_t h = FIXED;
        for (unsigned char c : s) {
            uint64_t x = c + 1;
            h ^= x + FIXED + (h<<6) + (h>>2);
        }
        return (size_t)h;
    }
};

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int N = -1, a = -1;
    if (argc >= 3) {
        N = stoi(argv[1]);
        a = stoi(argv[2]);
    } else {
        cout << "Enter N and a (e.g., 5 2): " << flush;
        if (!(cin >> N >> a)) return 0;
    }
    if (N < 0 || a < 0 || a > N) { cout << "0\n"; return 0; }
    if (a == 0 || a == N) { cout << 1 << "\n"; return 0; }

    a = min(a, N - a);

    if ((N >= 60 && a >= 10) || (a >= 13 && N >= 40) || (N >= 80)) {
        print_asymptotic_approx(N, a);
        return 0;
    }

    vector<vector<cpp_int>> C(N+1, vector<cpp_int>(a+1, 0));
    for (int n = 0; n <= N; ++n) {
        C[n][0] = 1;
        for (int k = 1; k <= min(n, a); ++k) C[n][k] = C[n-1][k-1] + C[n-1][k];
    }

    string start(a+1, char(0));
    start[a] = char(N);
    unordered_map<string, cpp_int, StrHash> dp, nxt;
    dp.reserve(1024);
    dp.emplace(start, cpp_int(1));

    auto process_all = [&](const vector<pair<string, cpp_int>>& items, unsigned useT) {
        vector<unordered_map<string, cpp_int, StrHash>> locals(useT);
        for (unsigned t = 0; t < useT; ++t) locals[t].reserve(1024);

        auto worker = [&](unsigned tid) {
            vector<int> c(a+1), x(a+1, 0), newc(a+1);
            for (size_t idx = 0; idx < items.size(); ++idx) {
                const string& key = items[idx].first;
                const cpp_int& val = items[idx].second;
                for (int k = 0; k <= a; ++k) c[k] = (unsigned char)key[k];
                int cap1 = min(a, c[1]);
                for (int take1 = 0; take1 <= cap1; ++take1) {
                    if (((take1 + (int)idx) % (int)useT) != (int)tid) continue;
                    x[1] = take1;
                    cpp_int ways = cpp_int(1) * C[c[1]][x[1]];

                    function<void(int,int,const cpp_int&)> dfs = [&](int r, int left, const cpp_int& w) {
                        if (r == a) {
                            x[r] = left;
                            if (x[r] > c[r]) return;
                            cpp_int w2 = w * C[c[r]][x[r]];
                            newc[0] = c[0] + x[1];
                            for (int k = 1; k < a; ++k) newc[k] = c[k] - x[k] + x[k+1];
                            newc[a] = c[a] - x[a];
                            for (int k = 0; k <= a; ++k) if (newc[k] < 0) return;
                            string outkey(a+1, char(0));
                            for (int k = 0; k <= a; ++k) outkey[k] = char(newc[k]);
                            locals[tid][outkey] += w2 * val;
                            return;
                        }
                        int cap = min(left, c[r]);
                        for (int take = 0; take <= cap; ++take) {
                            x[r] = take;
                            cpp_int wnext = w * C[c[r]][x[r]];
                            dfs(r+1, left - take, wnext);
                        }
                    };

                    if (a == 1) {
                        newc[0] = c[0] + x[1];
                        newc[1] = c[1] - x[1];
                        if (newc[0] >= 0 && newc[1] >= 0) {
                            string outkey(2, char(0));
                            outkey[0] = char(newc[0]);
                            outkey[1] = char(newc[1]);
                            locals[tid][outkey] += ways * val;
                        }
                    } else {
                        dfs(2, a - x[1], ways);
                    }
                }
            }
        };

        vector<thread> threads;
        threads.reserve(useT);
        for (unsigned t = 0; t < useT; ++t) threads.emplace_back(worker, t);
        for (auto& th : threads) th.join();

        unordered_map<string, cpp_int, StrHash> merged;
        size_t hint = 0;
        for (auto& lm : locals) hint += lm.size();
        merged.reserve(hint + 16);
        for (auto& lm : locals) for (auto& kv : lm) merged[kv.first] += kv.second;
        return merged;
    };

    for (int col = 0; col < N; ++col) {
        vector<pair<string, cpp_int>> items;
        items.reserve(dp.size());
        for (auto &p : dp) items.emplace_back(p.first, p.second);
        if (items.empty()) { cout << 0 << "\n"; return 0; }
        unsigned useT = max(1u, hw_threads());
        nxt = process_all(items, useT);
        dp.swap(nxt);
    }

    string finish(a+1, char(0));
    finish[0] = char(N);
    auto it = dp.find(finish);
    if (it == dp.end()) {
        cout << 0 << "\n";
    } else {
        cout << it->second << "\n";
    }
    return 0;
}