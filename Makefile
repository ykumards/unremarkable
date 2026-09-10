CXX ?= c++
CXXFLAGS ?= -O2 -std=c++23 -Wall -Wextra -Wpedantic -ffp-contract=off
CPPFLAGS ?= -Isrc -D_FILE_OFFSET_BITS=64
LDLIBS = -lm -lpthread
PYTHON ?= python3
SOURCES = src/main.cpp src/engine.cpp src/kernels.cpp src/pool.cpp src/tokenizer.cpp
HEADERS = src/engine.h src/kernels.h src/pool.h src/tokenizer.h
CORE = src/engine.cpp src/kernels.cpp src/pool.cpp src/tokenizer.cpp

.PHONY: all test sanitize models test-models chat-model chat-model-q8 tablet tablet-image
all: build/unremarkable

build:
	mkdir -p build

build/unremarkable: $(SOURCES) $(HEADERS) Makefile | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(SOURCES) $(LDFLAGS) $(LDLIBS) -o $@

build/unremarkable-sanitize: $(SOURCES) $(HEADERS) Makefile | build
	$(CXX) $(CPPFLAGS) -O1 -g -std=c++23 -Wall -Wextra -ffp-contract=off \
		-fsanitize=address,undefined -fno-omit-frame-pointer $(SOURCES) $(LDLIBS) -o $@

models:
	$(PYTHON) tools/download.py 15M

test-models:
	$(PYTHON) tools/download.py 260K

# SmolLM2-135M-Instruct: ~270 MB download, 538 MB FP32 checkpoint.
chat-model:
	$(PYTHON) tools/download.py smollm2
	$(PYTHON) tools/export_hf.py models/hf/SmolLM2-135M-Instruct \
		-o models/smollm2-135m.bin -c 2048

# The same weights as Q8_0: 144 MiB rather than 513 MiB, which is what makes the
# model fit the tablet comfortably. Quantizing needs numpy; nothing else does.
chat-model-q8:
	$(PYTHON) tools/download.py smollm2
	$(PYTHON) tools/export_hf.py models/hf/SmolLM2-135M-Instruct \
		-o models/smollm2-135m-q8.bin -c 2048 -q q8_0

# Cross build for the reMarkable 2: ARMv7 hard-float, NEON/VFPv4. libstdc++ and
# libgcc are linked statically because the toolchain ships 6.0.33 while the
# tablet has 6.0.32. -ffp-contract=off matters more here than on the host: VFPv4
# has FMA, and contraction would change results against the reference fixture.
TABLET_IMAGE ?= unremarkable-armv7
TABLET_CXX = arm-linux-gnueabihf-g++
TABLET_FLAGS = -march=armv7-a -mtune=cortex-a7 -mfpu=neon-vfpv4 -mfloat-abi=hard \
	-O2 -std=c++23 -Wall -Wextra -Wpedantic -ffp-contract=off \
	-static-libstdc++ -static-libgcc
TABLET_HOST ?= remar
TABLET_DIR ?= /home/root/unremarkable

tablet-image:
	docker build -t $(TABLET_IMAGE) -f tools/Dockerfile.armv7 tools

build/unremarkable-armv7: $(SOURCES) $(HEADERS) Makefile tools/Dockerfile.armv7 | build
	docker run --rm -v "$(CURDIR)":/src -w /src $(TABLET_IMAGE) \
		$(TABLET_CXX) $(CPPFLAGS) $(TABLET_FLAGS) $(SOURCES) $(LDLIBS) -o $@

tablet: build/unremarkable-armv7

# Separate sampling/call-count profiling build. Keep normal timing uninstrumented.
build/profile-armv7: $(SOURCES) $(HEADERS) Makefile tools/Dockerfile.armv7 | build
	docker run --rm -v "$(CURDIR)":/src -w /src $(TABLET_IMAGE) \
		$(TABLET_CXX) $(CPPFLAGS) $(TABLET_FLAGS) -g -pg $(SOURCES) $(LDLIBS) -o $@

build/probe: tests/probe.cpp $(CORE) $(HEADERS) Makefile | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) tests/probe.cpp $(CORE) $(LDFLAGS) $(LDLIBS) -o $@

build/probe-sanitize: tests/probe.cpp $(CORE) $(HEADERS) Makefile | build
	$(CXX) $(CPPFLAGS) -O1 -g -std=c++23 -Wall -Wextra -ffp-contract=off \
		-fsanitize=address,undefined -fno-omit-frame-pointer tests/probe.cpp $(CORE) $(LDLIBS) -o $@

build/test-matvec: tests/matvec.cpp src/kernels.cpp src/kernels.h Makefile | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) tests/matvec.cpp src/kernels.cpp $(LDFLAGS) $(LDLIBS) -o $@

build/test-matvec-sanitize: tests/matvec.cpp src/kernels.cpp src/kernels.h Makefile | build
	$(CXX) $(CPPFLAGS) -O1 -g -std=c++23 -ffp-contract=off \
		-fsanitize=address,undefined tests/matvec.cpp src/kernels.cpp $(LDLIBS) -o $@

test: build/unremarkable build/probe build/test-matvec
	./build/test-matvec
	$(PYTHON) tests/test_engine.py --binary build/unremarkable --probe build/probe

sanitize: build/unremarkable-sanitize build/probe-sanitize build/test-matvec-sanitize
	./build/test-matvec-sanitize
	$(PYTHON) tests/test_engine.py --binary build/unremarkable-sanitize --probe build/probe-sanitize

.PHONY: check check-sanitize format check-format
check: test
check-sanitize: sanitize
CLANG_FORMAT ?= $(firstword $(shell command -v clang-format 2>/dev/null) $(wildcard /opt/homebrew/opt/llvm@21/bin/clang-format /opt/homebrew/opt/llvm/bin/clang-format) clang-format)
format:
	$(CLANG_FORMAT) -i $(SOURCES) $(HEADERS) tests/probe.cpp tests/matvec.cpp
check-format:
	$(CLANG_FORMAT) --dry-run --Werror $(SOURCES) $(HEADERS) tests/probe.cpp tests/matvec.cpp
