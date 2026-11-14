#include "aquifer/portero.hpp"
#include "aquifer/morton.hpp"
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

int main() {
    std::cout << "=== Test Aquifer: Portero de HiperRAM (Ring-Buffer Pager) ===\n";

    const std::string tmp_file = "/tmp/aquifer_synth_weights.bin";
    const std::size_t num_layers = 28;
    const std::size_t layer_size = 4 * 1024 * 1024; // 4 MB por capa (total 112 MB en disco)

    // 1. Crear archivo sintético en disco
    std::cout << "  Creando archivo de prueba (" << (num_layers * layer_size) / (1024 * 1024) << " MB en disco)...\n";
    {
        std::ofstream f(tmp_file, std::ios::binary);
        std::vector<uint8_t> block(layer_size);
        for (std::size_t L = 0; L < num_layers; ++L) {
            std::fill(block.begin(), block.end(), static_cast<uint8_t>(L & 0xFF));
            f.write(reinterpret_cast<const char*>(block.data()), block.size());
        }
    }

    // 2. Inicializar el Portero con solo 2 slots de RAM
    std::cout << "  Inicializando Portero con 2 slots de RAM...\n";
    aquifer::Portero portero(tmp_file, 2);

    for (std::size_t L = 0; L < num_layers; ++L) {
        // Asignar una coordenada z normalizada en [-1, 1] a cada capa
        const double z = (double(L) / double(num_layers - 1)) * 2.0 - 1.0;
        const uint64_t morton = aquifer::cube_to_morton(0.0, 0.0, z);
        portero.register_layer(L, L * layer_size, layer_size, morton);
    }

    const std::size_t ram_used = portero.ram_bytes_used();
    std::cout << "  RAM fisica ocupada por el Portero: " << ram_used / (1024 * 1024)
              << " MB (vs " << (num_layers * layer_size) / (1024 * 1024) << " MB del modelo completo)\n";
    assert(ram_used == 2 * layer_size); // ¡Exactamente 8 MB de RAM en vez de 112 MB!

    // 3. Simular una pasada completa de inferencia token por token
    std::cout << "  Ejecutando pasada de inferencia capa por capa con prefetch asincrono...\n";
    auto t0 = std::chrono::high_resolution_clock::now();

    // Precargar la primera capa
    portero.prefetch_async(0);

    for (std::size_t L = 0; L < num_layers; ++L) {
        // Disparar prefetch de la siguiente capa en segundo plano
        if (L + 1 < num_layers) {
            portero.prefetch_async(L + 1);
        }

        // Adquirir la capa actual
        const uint8_t* ptr = portero.acquire(L);
        assert(ptr != nullptr);

        // Verificar integridad del primer y ultimo byte de la capa
        assert(ptr[0] == static_cast<uint8_t>(L & 0xFF));
        assert(ptr[layer_size - 1] == static_cast<uint8_t>(L & 0xFF));

        // Simular cálculo de GEMV (2 ms de computo)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "  Tiempo total para 28 capas: " << total_ms << " ms\n";
    std::cout << "  Ahorro de memoria: "
              << (100.0 * (1.0 - double(ram_used) / double(num_layers * layer_size))) << " %\n";

    ::unlink(tmp_file.c_str());
    std::cout << "=== TEST PORTERO PASO CON 100% DE ÉXITO ===\n";
    return 0;
}
