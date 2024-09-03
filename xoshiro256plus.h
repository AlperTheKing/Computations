#ifndef XOSHIRO256PLUS_H
#define XOSHIRO256PLUS_H

#include <cstdint>
#include <random>

class xoshiro256plus {
public:
    xoshiro256plus(uint64_t seed1, uint64_t seed2, uint64_t seed3, uint64_t seed4) {
        s[0] = seed1;
        s[1] = seed2;
        s[2] = seed3;
        s[3] = seed4;
    }

    uint64_t operator()() {
        const uint64_t result = s[0] + s[3];

        const uint64_t t = s[1] << 17;

        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];

        s[2] ^= t;

        s[3] = rotl(s[3], 45);

        return result;
    }

private:
    uint64_t s[4];

    static uint64_t rotl(const uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }
};

#endif // XOSHIRO256PLUS_H