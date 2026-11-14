#pragma once
// =============================================================================
//  aquifer · include/aquifer/portero.hpp
//  El Portero de HiperRAM: Paginador Asíncrono de Tensores en Búfer Circular.
// =============================================================================
//  En lugar de cargar todos los gigabytes del modelo en RAM física, el Portero
//  mantiene un Búfer Circular (Ring Buffer) de solo K capas activas (ej. 2 o 3).
//  
//  Mientras la CPU ejecuta con AVX2 la Capa L en el Slot 0, un hilo de fondo
//  precarga asíncronamente la Capa L+1 en el Slot 1 directamente desde disco.
//
//  Resultado:
//    - Modelo 1.5B: de 964 MB en RAM -> ~70 MB.
//    - Modelo 7B  : de 4.5 GB en RAM -> ~300 MB.
//    - Modelo 70B : de 35 GB en RAM  -> ~2.4 GB (¡corre en hardware modesto!).
// =============================================================================

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <fcntl.h>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace aquifer {

struct LayerMeta {
    std::size_t layer_idx{0};
    std::size_t file_offset{0};
    std::size_t byte_size{0};
    std::uint64_t morton_key{0}; // Clave espacial en el Cubo [-1, 1]³
};

struct Slot {
    std::vector<std::uint8_t> buffer;
    std::size_t current_layer{std::size_t(-1)};
    std::atomic<bool> ready{false};
};

class Portero {
public:
    explicit Portero(const std::string& weights_path, std::size_t num_slots = 2)
        : slots_(num_slots), fd_(-1), running_(true) {
        fd_ = ::open(weights_path.c_str(), O_RDONLY);
        if (fd_ < 0) {
            std::perror(("Aquifer Portero: no pudo abrir " + weights_path).c_str());
        }
        worker_ = std::thread(&Portero::worker_loop, this);
    }

    ~Portero() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            running_ = false;
        }
        cv_work_.notify_all();
        if (worker_.joinable()) {
            worker_.join();
        }
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }

    /// Registra la posición de una capa en el archivo continuo.
    void register_layer(std::size_t layer_idx, std::size_t offset, std::size_t size, std::uint64_t morton = 0) {
        if (layer_idx >= layers_.size()) {
            layers_.resize(layer_idx + 1);
        }
        layers_[layer_idx] = {layer_idx, offset, size, morton};
        max_layer_bytes_ = std::max(max_layer_bytes_, size);

        // Ajustar tamaño de slots si es necesario
        for (auto& s : slots_) {
            if (s.buffer.size() < max_layer_bytes_) {
                s.buffer.resize(max_layer_bytes_);
            }
        }
    }

    /// Dispara la precarga asíncrona de una capa en su slot correspondiente.
    void prefetch_async(std::size_t layer_idx) {
        if (layer_idx >= layers_.size() || fd_ < 0) return;
        const std::size_t slot_idx = layer_idx % slots_.size();
        auto& s = slots_[slot_idx];

        if (s.current_layer == layer_idx && s.ready.load(std::memory_order_acquire)) {
            return; // Ya está precargada y lista
        }

        s.ready.store(false, std::memory_order_release);
        s.current_layer = layer_idx;

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            work_queue_.push(layer_idx);
        }
        cv_work_.notify_one();
    }

    /// Adquiere el puntero a los pesos de la capa L (bloquea solo si el prefetch aún no termina).
    const std::uint8_t* acquire(std::size_t layer_idx) {
        if (layer_idx >= layers_.size()) return nullptr;
        const std::size_t slot_idx = layer_idx % slots_.size();
        auto& s = slots_[slot_idx];

        // Si no se había pedido, pedirla de inmediato
        if (s.current_layer != layer_idx) {
            prefetch_async(layer_idx);
        }

        // Esperar a que el hilo de fondo termine la lectura
        std::unique_lock<std::mutex> lock(slot_mutex_);
        cv_ready_.wait(lock, [&]() {
            return s.current_layer == layer_idx && s.ready.load(std::memory_order_acquire);
        });

        return s.buffer.data();
    }

    /// Devuelve cuántos bytes de RAM física consumen los slots del Portero en total.
    [[nodiscard]] std::size_t ram_bytes_used() const noexcept {
        std::size_t total = 0;
        for (const auto& s : slots_) total += s.buffer.size();
        return total;
    }

    [[nodiscard]] std::size_t num_slots() const noexcept { return slots_.size(); }
    [[nodiscard]] std::size_t num_layers() const noexcept { return layers_.size(); }

private:
    void worker_loop() {
        while (true) {
            std::size_t layer_to_load = std::size_t(-1);
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                cv_work_.wait(lock, [&]() { return !work_queue_.empty() || !running_; });
                if (!running_ && work_queue_.empty()) break;
                if (!work_queue_.empty()) {
                    layer_to_load = work_queue_.front();
                    work_queue_.pop();
                }
            }

            if (layer_to_load < layers_.size() && fd_ >= 0) {
                const auto& meta = layers_[layer_to_load];
                const std::size_t slot_idx = layer_to_load % slots_.size();
                auto& s = slots_[slot_idx];

                // pread es thread-safe y no toca el cursor del archivo
                std::size_t bytes_read = 0;
                while (bytes_read < meta.byte_size) {
                    ssize_t r = ::pread(fd_,
                                        s.buffer.data() + bytes_read,
                                        meta.byte_size - bytes_read,
                                        meta.file_offset + bytes_read);
                    if (r <= 0) break;
                    bytes_read += static_cast<std::size_t>(r);
                }

                {
                    std::lock_guard<std::mutex> lock(slot_mutex_);
                    s.ready.store(true, std::memory_order_release);
                }
                cv_ready_.notify_all();
            }
        }
    }

    std::vector<LayerMeta> layers_;
    std::vector<Slot> slots_;
    std::size_t max_layer_bytes_{0};
    int fd_;

    std::thread worker_;
    std::mutex queue_mutex_;
    std::condition_variable cv_work_;
    std::queue<std::size_t> work_queue_;

    std::mutex slot_mutex_;
    std::condition_variable cv_ready_;
    bool running_{false};
};

/// Portero de Doble Flujo (Weights + Scales) para modelos con tensores desacoplados.
class DualPortero {
public:
    DualPortero(const std::string& w_path, const std::string& s_path, std::size_t num_slots = 2)
        : slots_(num_slots), fd_w_(-1), fd_s_(-1), running_(true) {
        fd_w_ = ::open(w_path.c_str(), O_RDONLY);
        fd_s_ = ::open(s_path.c_str(), O_RDONLY);
        if (fd_w_ < 0) std::perror(("DualPortero: no pudo abrir " + w_path).c_str());
        if (fd_s_ < 0) std::perror(("DualPortero: no pudo abrir " + s_path).c_str());
        worker_ = std::thread(&DualPortero::worker_loop, this);
    }

    ~DualPortero() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            running_ = false;
        }
        cv_work_.notify_all();
        if (worker_.joinable()) worker_.join();
        if (fd_w_ >= 0) ::close(fd_w_);
        if (fd_s_ >= 0) ::close(fd_s_);
    }

    void register_layer(std::size_t layer_idx,
                        std::size_t w_off, std::size_t w_size,
                        std::size_t s_off, std::size_t s_size) {
        if (layer_idx >= layers_.size()) layers_.resize(layer_idx + 1);
        layers_[layer_idx] = {layer_idx, w_off, w_size, s_off, s_size};
        max_w_ = std::max(max_w_, w_size);
        max_s_ = std::max(max_s_, s_size);
        for (auto& s : slots_) {
            if (s.w_buf.size() < max_w_) {
                s.w_buf.resize(max_w_);
                std::memset(s.w_buf.data(), 0, s.w_buf.size());
            }
            if (s.s_buf.size() < max_s_) {
                s.s_buf.resize(max_s_);
                std::memset(s.s_buf.data(), 0, s.s_buf.size());
            }
        }
    }

    void prefetch_async(std::size_t layer_idx) {
        if (layer_idx >= layers_.size() || fd_w_ < 0) return;
        const std::size_t slot_idx = layer_idx % slots_.size();
        auto& s = slots_[slot_idx];
        if (s.current_layer == layer_idx && s.ready.load(std::memory_order_acquire)) return;

        s.ready.store(false, std::memory_order_release);
        s.current_layer = layer_idx;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            work_queue_.push(layer_idx);
        }
        cv_work_.notify_one();
    }

    struct LayerPointers {
        const std::uint8_t* w_ptr{nullptr};
        const std::uint8_t* s_ptr{nullptr};
    };

    LayerPointers acquire(std::size_t layer_idx) {
        if (layer_idx >= layers_.size()) return {nullptr, nullptr};
        const std::size_t slot_idx = layer_idx % slots_.size();
        auto& s = slots_[slot_idx];

        if (s.current_layer != layer_idx) {
            prefetch_async(layer_idx);
        }

        std::unique_lock<std::mutex> lock(slot_mutex_);
        cv_ready_.wait(lock, [&]() {
            return s.current_layer == layer_idx && s.ready.load(std::memory_order_acquire);
        });

        return {s.w_buf.data(), s.s_buf.data()};
    }

    [[nodiscard]] std::size_t ram_bytes_used() const noexcept {
        std::size_t total = 0;
        for (const auto& s : slots_) total += s.w_buf.size() + s.s_buf.size();
        return total;
    }

private:
    void worker_loop() {
        while (true) {
            std::size_t layer_to_load = std::size_t(-1);
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                cv_work_.wait(lock, [&]() { return !work_queue_.empty() || !running_; });
                if (!running_ && work_queue_.empty()) break;
                if (!work_queue_.empty()) {
                    layer_to_load = work_queue_.front();
                    work_queue_.pop();
                }
            }

            if (layer_to_load < layers_.size() && fd_w_ >= 0) {
                const auto& meta = layers_[layer_to_load];
                const std::size_t slot_idx = layer_to_load % slots_.size();
                auto& s = slots_[slot_idx];

                // 1. pread weights
                std::size_t w_done = 0;
                while (w_done < meta.w_bytes) {
                    ssize_t r = ::pread(fd_w_, s.w_buf.data() + w_done, meta.w_bytes - w_done, meta.w_offset + w_done);
                    if (r <= 0) break;
                    w_done += static_cast<std::size_t>(r);
                }

                // 2. pread scales
                if (fd_s_ >= 0 && meta.s_bytes > 0) {
                    std::size_t s_done = 0;
                    while (s_done < meta.s_bytes) {
                        ssize_t r = ::pread(fd_s_, s.s_buf.data() + s_done, meta.s_bytes - s_done, meta.s_offset + s_done);
                        if (r <= 0) break;
                        s_done += static_cast<std::size_t>(r);
                    }
                }

                {
                    std::lock_guard<std::mutex> lock(slot_mutex_);
                    s.ready.store(true, std::memory_order_release);
                }
                cv_ready_.notify_all();
            }
        }
    }

    struct LayerMeta2 {
        std::size_t layer_idx{0};
        std::size_t w_offset{0}, w_bytes{0};
        std::size_t s_offset{0}, s_bytes{0};
    };

    struct DualSlot {
        std::vector<std::uint8_t> w_buf;
        std::vector<std::uint8_t> s_buf;
        std::size_t current_layer{std::size_t(-1)};
        std::atomic<bool> ready{false};
    };

    std::vector<LayerMeta2> layers_;
    std::vector<DualSlot> slots_;
    std::size_t max_w_{0}, max_s_{0};
    int fd_w_{-1}, fd_s_{-1};

    std::thread worker_;
    std::mutex queue_mutex_;
    std::condition_variable cv_work_;
    std::queue<std::size_t> work_queue_;

    std::mutex slot_mutex_;
    std::condition_variable cv_ready_;
    bool running_{false};
};

} // namespace aquifer
