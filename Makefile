CXX = g++
CXXFLAGS = -std=c++23 -O3 -mavx2 -mfma -mf16c -mbmi2 -Wall -Wextra -Wpedantic -pthread -I include
CXXFLAGS_SSE = -std=c++23 -O3 -msse4.2 -Wall -Wextra -Wpedantic -pthread -I include

BUILD_DIR = build
TESTS = $(BUILD_DIR)/test_morton $(BUILD_DIR)/test_portero $(BUILD_DIR)/test_paging_rss \
        $(BUILD_DIR)/test_sse_affine $(BUILD_DIR)/test_affine_g32 $(BUILD_DIR)/test_kmin $(BUILD_DIR)/test_affine_g64 \
        $(BUILD_DIR)/test_doublemap $(BUILD_DIR)/check_g32M
BINS = $(BUILD_DIR)/generate_aquifer $(BUILD_DIR)/generate_aquifer_sse $(BUILD_DIR)/generate_aquifer_diamond

.PHONY: all clean test tests bins sse

all: tests bins

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/test_morton: tests/test_morton.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BUILD_DIR)/test_portero: tests/test_portero.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BUILD_DIR)/test_paging_rss: tests/test_paging_rss.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BUILD_DIR)/test_sse_affine: tests/test_sse_affine.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_SSE) $< -o $@

$(BUILD_DIR)/test_affine_g32: tests/test_affine_g32.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BUILD_DIR)/test_kmin: tests/test_kmin.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BUILD_DIR)/test_affine_g64: tests/test_affine_g64.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BUILD_DIR)/test_doublemap: tests/test_doublemap.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BUILD_DIR)/check_g32M: tests/check_g32M.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BUILD_DIR)/generate_aquifer: tools/generate_aquifer.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BUILD_DIR)/generate_aquifer_sse: tools/generate_aquifer.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_SSE) $< -o $@

$(BUILD_DIR)/generate_aquifer_diamond: tools/generate_aquifer_diamond.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

sse: $(BUILD_DIR)/generate_aquifer_sse

tests: $(TESTS)

bins: $(BINS)

test: tests
	@echo "=== Ejecutando Suite de Tests Aquifer ==="
	@$(BUILD_DIR)/test_morton
	@echo ""
	@$(BUILD_DIR)/test_portero
	@echo ""
	@$(BUILD_DIR)/test_paging_rss
	@echo ""
	@$(BUILD_DIR)/test_sse_affine
	@echo ""
	@$(BUILD_DIR)/test_affine_g32
	@echo ""
	@$(BUILD_DIR)/test_kmin
	@echo ""
	@$(BUILD_DIR)/test_affine_g64
	@echo ""
	@$(BUILD_DIR)/test_doublemap
	@echo ""
	@$(BUILD_DIR)/check_g32M
	@echo ""
	@echo ">>> TODOS LOS TESTS DE AQUIFER PASARON CON ÉXITO <<<"

clean:
	rm -f $(TESTS) $(BINS)
