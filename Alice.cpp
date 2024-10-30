#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <limits>
#include <unordered_map>
#include <gmp.h>
#include <gmpxx.h>

using namespace std;

typedef mpz_class ll;

const ll MOD = 1000000007;

ll mod(const ll& a) {
    ll res = a % MOD;
    return (res < 0) ? res + MOD : res;
}

ll modinv(const ll& a, const ll& m) {
    ll r;
    if (mpz_invert(r.get_mpz_t(), a.get_mpz_t(), m.get_mpz_t()) == 0) {
        return 0; // Inverse does not exist
    }
    return r;
}

ll safe_invert(ll num, ll den) {
    if (den == 0) throw invalid_argument("Zero denominator");
    ll inv_den = modinv(den, MOD);
    return mod(num * inv_den);
}

struct Transformation {
    ll a, b, c, d, e, f; // x' = a*x + b*y + e, y' = c*x + d*y + f
    Transformation() : a(1), b(0), c(0), d(1), e(0), f(0) {}
};

struct TestCase {
    int K;
    ll N;
    string X0_s, Y0_s;
    vector<string> commands;
};

void read_test_case(TestCase& tc) {
    cin >> tc.K;
    string N_str;
    cin >> N_str >> tc.X0_s >> tc.Y0_s;
    tc.N = ll(N_str);

    cin.ignore(numeric_limits<streamsize>::max(), '\n');
    for (int i = 0; i < tc.K; ++i) {
        string line;
        getline(cin, line);
        while (line.empty())
            getline(cin, line);
        tc.commands.push_back(line);
    }
}

ll parse_fraction(const string& s) {
    size_t pos = s.find('/');
    ll num(s.substr(0, pos));
    ll den = (pos != string::npos) ? ll(s.substr(pos + 1)) : 1;
    if (den == 0) throw invalid_argument("Invalid denominator");
    ll inv_den = modinv(den, MOD);
    if (inv_den == 0) throw invalid_argument("No modular inverse");
    return mod(num * inv_den);
}

void process_test_case(const TestCase& tc) {
    ll X0, Y0;
    try {
        X0 = parse_fraction(tc.X0_s);
        Y0 = parse_fraction(tc.Y0_s);
    } catch (...) {
        cout << "WONDERLAND" << endl;
        return;
    }

    vector<pair<Transformation, bool>> sequence;

    for (const string& cmd_line : tc.commands) {
        stringstream ss(cmd_line);
        string cmd;
        ss >> cmd;
        if (cmd == "S") {
            string c_s;
            ss >> c_s;
            ll c = parse_fraction(c_s);
            Transformation t;
            t.a = c;
            t.d = c;
            sequence.push_back({ t, false });
        } else if (cmd == "T") {
            string a_s, b_s;
            ss >> a_s >> b_s;
            ll a = parse_fraction(a_s);
            ll b = parse_fraction(b_s);
            Transformation t;
            t.e = a;
            t.f = b;
            sequence.push_back({ t, false });
        } else if (cmd == "R") {
            string a_s, b_s;
            ss >> a_s >> b_s;
            ll a = parse_fraction(a_s);
            ll b = parse_fraction(b_s);
            Transformation t;
            t.a = a; t.b = mod(-b);
            t.c = b; t.d = a;
            sequence.push_back({ t, false });
        } else if (cmd == "F") {
            string axis;
            ss >> axis;
            Transformation t;
            if (axis == "X") {
                t.d = mod(-1);
            } else if (axis == "Y") {
                t.a = mod(-1);
            } else {
                cout << "WONDERLAND" << endl;
                return;
            }
            sequence.push_back({ t, false });
        } else if (cmd == "I") {
            sequence.push_back({ Transformation(), true });
        } else {
            cout << "WONDERLAND" << endl;
            return;
        }
    }

    unordered_map<string, size_t> state_map;
    vector<tuple<ll, ll, int>> states;

    ll x = X0;
    ll y = Y0;
    int inversion_state = 0;

    size_t cycle_start = -1;
    size_t cycle_length = -1;

    for (size_t step = 0; step < tc.N.get_ui(); ++step) {
        for (auto& p : sequence) {
            if (p.second) { // Inversion
                ll denom = mod(x * x + y * y);
                if (denom == 0) {
                    cout << "WONDERLAND" << endl;
                    return;
                }
                x = safe_invert(x, denom);
                y = safe_invert(y, denom);
                inversion_state ^= 1;
            } else { // Transformation
                Transformation& t = p.first;
                ll new_x = mod(t.a * x + t.b * y + t.e);
                ll new_y = mod(t.c * x + t.d * y + t.f);
                x = new_x;
                y = new_y;
            }
        }

        string key = x.get_str() + "," + y.get_str() + "," + to_string(inversion_state);
        if (state_map.count(key)) {
            cycle_start = state_map[key];
            cycle_length = step - cycle_start;
            break;
        } else {
            state_map[key] = step;
            states.push_back({ x, y, inversion_state });
        }
    }

    if (cycle_length != static_cast<size_t>(-1)) {
        mpz_class remaining = (tc.N - cycle_start) % cycle_length;
        size_t index = cycle_start + remaining.get_ui();
        if (index >= states.size()) {
            cout << "WONDERLAND" << endl;
            return;
        }
        auto& final_state = states[index];
        x = get<0>(final_state);
        y = get<1>(final_state);
        inversion_state = get<2>(final_state);
    }

    if (inversion_state % 2 == 1) {
        ll denom = mod(x * x + y * y);
        if (denom == 0) {
            cout << "WONDERLAND" << endl;
            return;
        }
        x = safe_invert(x, denom);
        y = safe_invert(y, denom);
    }

    cout << x.get_ui() << " " << y.get_ui() << endl;
}

int main() {
    int T;
    cin >> T;
    vector<TestCase> test_cases(T);

    for (int t = 0; t < T; ++t) {
        read_test_case(test_cases[t]);
    }

    for (int t = 0; t < T; ++t) {
        process_test_case(test_cases[t]);
    }

    return 0;
}