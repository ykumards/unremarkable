CXX ?= c++
CXXFLAGS ?= -O2 -std=c++23 -Wall -Wextra -Wpedantic -ffp-contract=off
CPPFLAGS ?= -Isrc -D_FILE_OFFSET_BITS=64
LDLIBS = -lm -pthread
PYTHON ?= python3
SOURCES = src/main.cpp src/engine.cpp src/prefill.cpp src/model.cpp src/sampler.cpp src/worker.cpp src/kernels.cpp src/tokenizer.cpp
HEADERS = src/engine.h src/model.h src/profile.h src/sampler.h src/worker.h src/kernels.h src/tokenizer.h
CORE = src/engine.cpp src/prefill.cpp src/model.cpp src/sampler.cpp src/worker.cpp src/kernels.cpp src/tokenizer.cpp

.PHONY: all test sanitize models test-models chat-model chat-model-q8 accuracy tablet tablet-profile tablet-image
all: build/unremarkable

build:
	mkdir -p build

build/unremarkable: $(SOURCES) $(HEADERS) Makefile | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(SOURCES) $(LDFLAGS) $(LDLIBS) -o $@

build/unremarkable-sanitize: $(SOURCES) $(HEADERS) Makefile | build
	$(CXX) $(CPPFLAGS) -O1 -g -std=c++23 -Wall -Wextra -ffp-contract=off \
		-fsanitize=address,undefined -fno-omit-frame-pointer $(SOURCES) $(LDLIBS) -o $@

# Per-step timings on a second stderr line; see docs/development.md.
build/unremarkable-profile: $(SOURCES) $(HEADERS) Makefile | build
	$(CXX) $(CPPFLAGS) -DUNREMARKABLE_PROFILE $(CXXFLAGS) $(SOURCES) $(LDFLAGS) $(LDLIBS) -o $@

models:
	$(PYTHON) tools/download.py 15M

test-models:
	$(PYTHON) tools/download.py 260K

# SmolLM2-135M-Instruct: ~270 MB download, 538 MB FP32 checkpoint.
chat-model:
	$(PYTHON) tools/download.py smollm2
	$(PYTHON) tools/export_hf.py models/hf/SmolLM2-135M-Instruct \
		-o models/smollm2-135m.bin -c 2048

# The same weights as Q8_0: 144 MiB rather than 513 MiB. Quantizing needs numpy.
chat-model-q8:
	$(PYTHON) tools/download.py smollm2
	$(PYTHON) tools/export_hf.py models/hf/SmolLM2-135M-Instruct \
		-o models/smollm2-135m-q8.bin -c 2048 -q q8_0

# Q8_0 against FP32 on the first 40 chunks of WikiText-2 test, fixed in advance.
accuracy: build/perplexity
	$(PYTHON) tools/download.py wikitext
	./build/perplexity models/smollm2-135m.bin models/smollm2-135m-q8.bin \
		models/smollm2-135m.tok models/wikitext-2-raw/wiki.test.raw 40

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

build/unremarkable-armv7-profile: $(SOURCES) $(HEADERS) Makefile tools/Dockerfile.armv7 | build
	docker run --rm -v "$(CURDIR)":/src -w /src $(TABLET_IMAGE) \
		$(TABLET_CXX) $(CPPFLAGS) -DUNREMARKABLE_PROFILE $(TABLET_FLAGS) $(SOURCES) $(LDLIBS) -o $@

tablet-profile: build/unremarkable-armv7-profile

build/probe: tests/probe.cpp $(CORE) $(HEADERS) Makefile | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) tests/probe.cpp $(CORE) $(LDFLAGS) $(LDLIBS) -o $@

build/perplexity: tools/perplexity.cpp $(CORE) $(HEADERS) Makefile | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) tools/perplexity.cpp $(CORE) $(LDFLAGS) $(LDLIBS) -o $@

build/probe-sanitize: tests/probe.cpp $(CORE) $(HEADERS) Makefile | build
	$(CXX) $(CPPFLAGS) -O1 -g -std=c++23 -Wall -Wextra -ffp-contract=off \
		-fsanitize=address,undefined -fno-omit-frame-pointer tests/probe.cpp $(CORE) $(LDLIBS) -o $@

test: build/unremarkable build/unremarkable-profile build/probe build/test-prefill build/test-worker build/test-q8 build/test-q8-scalar build/test-attention build/test-attention-scalar
	./build/test-attention
	./build/test-attention-scalar
	./build/test-q8
	./build/test-q8-scalar
	./build/test-worker
	$(PYTHON) tests/test_engine.py --binary build/unremarkable --probe build/probe --prefill build/test-prefill \
		--profile-binary build/unremarkable-profile

sanitize: build/unremarkable-sanitize build/probe-sanitize build/test-prefill-sanitize build/test-worker-sanitize build/test-q8-sanitize build/test-attention-sanitize
	./build/test-attention-sanitize
	./build/test-q8-sanitize
	./build/test-worker-sanitize
	$(PYTHON) tests/test_engine.py --binary build/unremarkable-sanitize --probe build/probe-sanitize --prefill build/test-prefill-sanitize

build/test-attention: tests/attention.cpp src/kernels.cpp src/kernels.h Makefile | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) tests/attention.cpp src/kernels.cpp $(LDLIBS) -o $@

build/test-attention-scalar: tests/attention.cpp src/kernels.cpp src/kernels.h Makefile | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -DUNREMARKABLE_SCALAR tests/attention.cpp src/kernels.cpp $(LDLIBS) -o $@

build/test-attention-sanitize: tests/attention.cpp src/kernels.cpp src/kernels.h Makefile | build
	$(CXX) $(CPPFLAGS) -O1 -g -std=c++23 -ffp-contract=off -fsanitize=address,undefined \
		-fno-omit-frame-pointer tests/attention.cpp src/kernels.cpp $(LDLIBS) -o $@

build/test-worker: tests/worker.cpp src/worker.cpp src/worker.h Makefile | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) tests/worker.cpp src/worker.cpp $(LDLIBS) -o $@

build/test-worker-sanitize: tests/worker.cpp src/worker.cpp src/worker.h Makefile | build
	$(CXX) $(CPPFLAGS) -O1 -g -std=c++23 -fsanitize=address,undefined \
		-fno-omit-frame-pointer tests/worker.cpp src/worker.cpp $(LDLIBS) -o $@

build/test-q8: tests/q8.cpp src/kernels.cpp src/kernels.h src/model.h Makefile | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) tests/q8.cpp src/kernels.cpp $(LDLIBS) -o $@

build/test-q8-scalar: tests/q8.cpp src/kernels.cpp src/kernels.h src/model.h Makefile | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -DUNREMARKABLE_SCALAR tests/q8.cpp src/kernels.cpp $(LDLIBS) -o $@

build/test-q8-sanitize: tests/q8.cpp src/kernels.cpp src/kernels.h src/model.h Makefile | build
	$(CXX) $(CPPFLAGS) -O1 -g -std=c++23 -ffp-contract=off -fsanitize=address,undefined \
		-fno-omit-frame-pointer tests/q8.cpp src/kernels.cpp $(LDLIBS) -o $@

build/test-prefill: tests/prefill.cpp $(CORE) $(HEADERS) Makefile | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) tests/prefill.cpp $(CORE) $(LDLIBS) -o $@

build/test-prefill-sanitize: tests/prefill.cpp $(CORE) $(HEADERS) Makefile | build
	$(CXX) $(CPPFLAGS) -O1 -g -std=c++23 -ffp-contract=off -fsanitize=address,undefined \
		-fno-omit-frame-pointer tests/prefill.cpp $(CORE) $(LDLIBS) -o $@

.PHONY: check check-sanitize format check-format
check: test
check-sanitize: sanitize
CLANG_FORMAT ?= $(firstword $(shell command -v clang-format 2>/dev/null) $(wildcard /opt/homebrew/opt/llvm@21/bin/clang-format /opt/homebrew/opt/llvm/bin/clang-format) clang-format)
format:
	$(CLANG_FORMAT) -i $(SOURCES) $(HEADERS) tests/probe.cpp tests/worker.cpp tests/q8.cpp tests/prefill.cpp tests/attention.cpp tools/perplexity.cpp
check-format:
	$(CLANG_FORMAT) --dry-run --Werror $(SOURCES) $(HEADERS) tests/probe.cpp tests/worker.cpp tests/q8.cpp tests/prefill.cpp tests/attention.cpp tools/perplexity.cpp
