#include "aquifer/morton.hpp"
#include <cassert>
#include <chrono>
#include <iostream>
#include <random>

int main() {
    std::cout << "=== Test Aquifer: Morton 3D Coordinate Indexer ===\n";

    // 1. Verificación en puntos canónicos del hipercubo
    const double canonic[8][3] = {
        {-1.0, -1.0, -1.0},
        {-1.0, -1.0,  1.0},
        {-1.0,  1.0, -1.0},
        {-1.0,  1.0,  1.0},
        { 1.0, -1.0, -1.0},
        { 1.0, -1.0,  1.0},
        { 1.0,  1.0, -1.0},
        { 1.0,  1.0,  1.0}
    };

    for (int i = 0; i < 8; ++i) {
        const double x = canonic[i][0], y = canonic[i][1], z = canonic[i][2];
        const uint64_t code = aquifer::cube_to_morton(x, y, z);
        double rx, ry, rz;
        aquifer::morton_to_cube(code, rx, ry, rz);
        assert(std::abs(x - rx) < 1e-5);
        assert(std::abs(y - ry) < 1e-5);
        assert(std::abs(z - rz) < 1e-5);
        std::cout << "  Esquina [" << x << ", " << y << ", " << z << "] -> Morton: 0x"
                  << std::hex << code << std::dec << "\n";
    }

    // 2. Roundtrip masivo aleatorio (100.000 puntos)
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    const int N = 100'000;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        const double x = dist(rng), y = dist(rng), z = dist(rng);
        const uint64_t code = aquifer::cube_to_morton(x, y, z);
        double rx, ry, rz;
        aquifer::morton_to_cube(code, rx, ry, rz);
        if (std::abs(x - rx) >= 1e-5 || std::abs(y - ry) >= 1e-5 || std::abs(z - rz) >= 1e-5) {
            std::cerr << "Discrepancia en punto " << x << " " << y << " " << z << "\n";
            return 1;
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ns_per_op = std::chrono::duration<double, std::nano>(t1 - t0).count() / N;

    std::cout << "  Roundtrip de 100.000 puntos en el Cubo OK\n";
    std::cout << "  Latencia media de indexacion espacial: " << ns_per_op << " ns/op\n";
    std::cout << "=== TODOS LOS TESTS MORTON PASARON CON ÉXITO ===\n";
    return 0;
}
