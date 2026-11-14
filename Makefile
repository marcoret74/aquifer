CXX = g++
CXXFLAGS = -std=c++23 -O3 -mavx2 -mfma -mf16c -mbmi2 -Wall -Wextra -Wpedantic -pthread -I include
CXXFLAGS_SSE = -std=c++23 -O3 -msse4.2 -Wall -Wextra -Wpedantic -pthread -I include

BUILD_DIR = build
TESTS = $(BUILD_DIR)/test_morton $(BUILD_DIR)/test_portero $(BUILD_DIR)/test_paging_rss $(BUILD_DIR)/test_sse_affine
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
	@echo ">>> TODOS LOS TESTS DE AQUIFER PASARON CON ÉXITO <<<"

clean:
	rm -f $(TESTS) $(BINS)
