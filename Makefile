CUDA_HOME ?= /usr/local/cuda
PYTHON ?= python3
BUILD_DIR ?= build
EXAMPLE_DIR ?= examples

FULL_LIB ?= $(BUILD_DIR)/libflash_attn_sm80.so
FWD_LIB ?= $(BUILD_DIR)/libflash_attn_sm80_fwd.so
MINIMAL_EXE ?= $(BUILD_DIR)/minimal_fwd
BENCH_EXE ?= $(BUILD_DIR)/bench_fwd

NVCC_THREADS ?= 4
JOBS ?= $(shell nproc 2>/dev/null || echo 8)
FAST_HEAD_DIMS ?= 64
FAST_DTYPES ?= fp16 bf16

TORCH_INCLUDE := $(shell $(PYTHON) -c "from torch.utils.cpp_extension import include_paths; print(' '.join(include_paths()))")
TORCH_LIB := $(shell $(PYTHON) -c "from torch.utils.cpp_extension import library_paths; print(library_paths()[0])")
TORCH_ABI := $(shell $(PYTHON) -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")

CXX := g++
NVCC := $(CUDA_HOME)/bin/nvcc

INCLUDES := -I./csrc/flash_attn/include -I./csrc/flash_attn/src/core -I./csrc/flash_attn/src/templates -I./csrc/flash_attn/src/generated -I./csrc/cutlass/include -I$(CUDA_HOME)/include $(addprefix -I,$(TORCH_INCLUDE))
CXXFLAGS := -O3 -std=c++17 -fPIC -MMD -MP -D_GLIBCXX_USE_CXX11_ABI=$(TORCH_ABI)
NVCCFLAGS_BASE := -O3 -std=c++17 --threads $(NVCC_THREADS) -Xcompiler -fPIC -D_GLIBCXX_USE_CXX11_ABI=$(TORCH_ABI) \
	-U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ -U__CUDA_NO_HALF2_OPERATORS__ \
	-U__CUDA_NO_BFLOAT16_CONVERSIONS__ --expt-relaxed-constexpr --expt-extended-lambda \
	--use_fast_math -gencode arch=compute_80,code=sm_80
SO_LDFLAGS := -shared -L$(TORCH_LIB) -L$(CUDA_HOME)/lib64 \
	-Wl,-rpath,$(TORCH_LIB) -Wl,-rpath,$(CUDA_HOME)/lib64
EXE_LDFLAGS := -L$(TORCH_LIB) -L$(CUDA_HOME)/lib64 -L$(BUILD_DIR) \
	-Wl,-rpath,$(TORCH_LIB) -Wl,-rpath,$(CUDA_HOME)/lib64 -Wl,-rpath,$(BUILD_DIR)
LDLIBS := -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -lcudart

FWD_CPP_OBJ := $(BUILD_DIR)/flash_api_fwd.o
FULL_CPP_OBJ := $(BUILD_DIR)/flash_api_full.o
MINIMAL_OBJ := $(BUILD_DIR)/minimal_fwd.o
BENCH_OBJ := $(BUILD_DIR)/bench_fwd.o

FWD_ALL_CU_SRCS := $(sort $(wildcard csrc/flash_attn/src/generated/flash_fwd*sm80.cu))
BWD_ALL_CU_SRCS := $(sort $(wildcard csrc/flash_attn/src/generated/flash_bwd*sm80.cu))

FAST_FWD_CU_SRCS := $(foreach d,$(FAST_HEAD_DIMS),$(foreach t,$(FAST_DTYPES), \
	$(wildcard csrc/flash_attn/src/generated/flash_fwd_hdim$(d)_$(t)_sm80.cu) \
	$(wildcard csrc/flash_attn/src/generated/flash_fwd_hdim$(d)_$(t)_causal_sm80.cu) \
	$(wildcard csrc/flash_attn/src/generated/flash_fwd_split_hdim$(d)_$(t)_sm80.cu) \
	$(wildcard csrc/flash_attn/src/generated/flash_fwd_split_hdim$(d)_$(t)_causal_sm80.cu)))

ifeq ($(strip $(FAST_FWD_CU_SRCS)),)
$(error FAST_FWD_CU_SRCS is empty. Check FAST_HEAD_DIMS='$(FAST_HEAD_DIMS)' and FAST_DTYPES='$(FAST_DTYPES)')
endif

FWD_FAST_CU_OBJS := $(patsubst csrc/flash_attn/src/generated/%.cu,$(BUILD_DIR)/fwd_fast_%.o,$(FAST_FWD_CU_SRCS))
FWD_ALL_CU_OBJS := $(patsubst csrc/flash_attn/src/generated/%.cu,$(BUILD_DIR)/full_%.o,$(FWD_ALL_CU_SRCS))
BWD_ALL_CU_OBJS := $(patsubst csrc/flash_attn/src/generated/%.cu,$(BUILD_DIR)/full_%.o,$(BWD_ALL_CU_SRCS))

.PHONY: all fast full lib fwd examples minimal bench clean print-config

all: fwd

lib: full

fwd: $(FWD_LIB)

full: $(FULL_LIB)

examples: minimal bench

minimal: $(MINIMAL_EXE)

bench: $(BENCH_EXE)

print-config:
	@echo "CUDA_HOME=$(CUDA_HOME)"
	@echo "TORCH_LIB=$(TORCH_LIB)"
	@echo "TORCH_ABI=$(TORCH_ABI)"
	@echo "JOBS=$(JOBS)"
	@echo "NVCC_THREADS=$(NVCC_THREADS)"
	@echo "FAST_HEAD_DIMS=$(FAST_HEAD_DIMS)"
	@echo "FAST_DTYPES=$(FAST_DTYPES)"
	@echo "FWD_LIB=$(FWD_LIB)"
	@echo "FULL_LIB=$(FULL_LIB)"

$(FWD_LIB): $(FWD_CPP_OBJ) $(FWD_FAST_CU_OBJS) | $(BUILD_DIR)
	$(NVCC) $(SO_LDFLAGS) -o $@ $^ $(LDLIBS)

$(FULL_LIB): $(FULL_CPP_OBJ) $(FWD_ALL_CU_OBJS) $(BWD_ALL_CU_OBJS) | $(BUILD_DIR)
	$(CXX) $(SO_LDFLAGS) -o $@ $^ $(LDLIBS)

$(MINIMAL_EXE): $(MINIMAL_OBJ) $(FWD_LIB) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(MINIMAL_OBJ) -lflash_attn_sm80_fwd $(EXE_LDFLAGS) $(LDLIBS)

$(BENCH_EXE): $(BENCH_OBJ) $(FWD_LIB) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(BENCH_OBJ) -lflash_attn_sm80_fwd $(EXE_LDFLAGS) $(LDLIBS)

$(FWD_CPP_OBJ): csrc/flash_attn/flash_api.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -DFLASHATTENTION_DISABLE_BACKWARD $(INCLUDES) -c $< -o $@

$(FULL_CPP_OBJ): csrc/flash_attn/flash_api.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(MINIMAL_OBJ): $(EXAMPLE_DIR)/minimal_fwd.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BENCH_OBJ): $(EXAMPLE_DIR)/bench_fwd.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/fwd_fast_%.o: csrc/flash_attn/src/generated/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS_BASE) -DFLASHATTENTION_DISABLE_BACKWARD $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/full_%.o: csrc/flash_attn/src/generated/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS_BASE) $(INCLUDES) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)

-include $(wildcard $(BUILD_DIR)/*.d)
