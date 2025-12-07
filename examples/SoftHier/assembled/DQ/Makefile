MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
CUR_DIR := $(MAKEFILE_DIR)
SH_DIR := $(realpath $(MAKEFILE_DIR)../../../..)
# reprocucinility, deactivate for speedup
# export CUBLAS_WORKSPACE_CONFIG := :4096:8
# export OMP_NUM_THREADS := 1
# export MKL_NUM_THREADS := 1
# export CUDA_LAUNCH_BLOCKING := 1
ARCH_FILE_DQ ?= $(CUR_DIR)config/arch/arch_summa.py
ifdef arch
	ARCH_FILE_DQ = $(realpath $(arch))
endif

hw:
	@echo "Compiling SoftHier HW & SW for attention ..."
	cfg=$(ARCH_FILE_DQ) make -C $(SH_DIR) hw
export PYTHONPATH := $(CUR_DIR)scripts:$(PYTHONPATH)
pld_gemm:
	cd software/util; python store_cb_and_idx.py -kernel gemm
pld_gemv:
	cd software/util; python store_cb_and_idx.py -kernel gemv
dq:
	DEBUG=1 cfg=$(ARCH_FILE_DQ) app=$(abspath $(CUR_DIR)software) make -C $(SH_DIR) hs run
dqsw:
	DEBUG=1 cfg=$(ARCH_FILE_DQ) app=$(abspath $(CUR_DIR)software) make -C $(SH_DIR) sw run
dq_nodebug:
	cfg=$(ARCH_FILE_DQ) app=$(abspath $(CUR_DIR)software) make -C $(SH_DIR) hs run
pfto:
	VERIFY_VALUES=0 DEBUG=1 cfg=$(ARCH_FILE_DQ) app=$(abspath $(CUR_DIR)software) make -C $(SH_DIR) hs runv pfto
rund:
	DEBUG=0 cfg=$(ARCH_FILE_DQ) app=$(abspath $(CUR_DIR)software) make -C $(SH_DIR) hs rund
runv:
	DEBUG=0 cfg=$(ARCH_FILE_DQ) app=$(abspath $(CUR_DIR)software) make -C $(SH_DIR) hs runv
dqswv:
	make pld_gemv;
	DEBUG=1 cfg=$(ARCH_FILE_DQ) app=$(abspath $(CUR_DIR)software) make -C $(SH_DIR) sw run
dqswm:
	make pld_gemm;
	DEBUG=1 cfg=$(ARCH_FILE_DQ) app=$(abspath $(CUR_DIR)software) make -C $(SH_DIR) sw run

# Test targets
test:
	DEBUG=1 cfg=$(ARCH_FILE_DQ) app=$(abspath $(CUR_DIR)software) make -C $(SH_DIR) hs
	$(SH_DIR)/install/bin/gvsoc --target=pulp.chips.flex_cluster.flex_cluster --binary $(SH_DIR)/sw_build/softhier_test.elf run

testsw:
	DEBUG=1 cfg=$(ARCH_FILE_DQ) app=$(abspath $(CUR_DIR)software) make -C $(SH_DIR) sw
	$(SH_DIR)/install/bin/gvsoc --target=pulp.chips.flex_cluster.flex_cluster --binary $(SH_DIR)/sw_build/softhier_test.elf run


# GEMM configuration
ARCH_FILE = ${CUR_DIR}/config/arch/arch_summa.py
GEMM_FILE ?= $(CUR_DIR)config/kernels/gemm.py
ifdef gemm
	GEMM_FILE = $(realpath $(gemm))
endif
GEMV_FILE ?= $(CUR_DIR)config/kernels/gemv.py
ifdef gemv
	GEMV_FILE = $(realpath $(gemv))
endif
gemm-cfg:
	@echo "Generating configuration files for GEMM ..."
	python3 ${CUR_DIR}/scripts/kernels/gemm_config.py ${GEMM_FILE} ${CUR_DIR}/software/SummaGEMM/include/gemm.h

gemm-pld:
	@echo "Generating preload files for GEMM ..."
	python3 ${CUR_DIR}/scripts/kernels/gemm_preload.py \
			${CUR_DIR}/software/SummaGEMM/preload.elf \
			${CUR_DIR}/software/SummaGEMM/include/gemm_addresses.h \
			${ARCH_FILE} \
			${GEMM_FILE}

gemm-data:
	@echo "Generating data header for GEMM (linker script-based) ..."
	python3 ${CUR_DIR}/scripts/kernels/gemm_data_generator.py \
			${CUR_DIR}/software/SummaGEMM/include/gemm_data.h \
			${ARCH_FILE} \
			${GEMM_FILE}

gemm-hs:
	@echo "Compiling SoftHier HW & software for GEMM ..."
	cfg=${ARCH_FILE} app=${CUR_DIR}/software/SummaGEMM make -C ../../../../ hs

gemm-sw:gemm-pld
	@echo "Compiling SoftHier software for GEMM ..."
	cfg=${ARCH_FILE} app=${CUR_DIR}/software/SummaGEMM make -C ${SH_DIR} sw
	@echo "Running SoftHier simulation for GEMM ..."
	pld=${CUR_DIR}/software/SummaGEMM/preload.elf cfg=${ARCH_FILE} app=${CUR_DIR}/software/SummaGEMM make -C ${SH_DIR} run
gemm-swd:gemm-pld
	@echo "Compiling SoftHier software for GEMM ..."
	cfg=${ARCH_FILE} app=${CUR_DIR}/software/SummaGEMM make -C ${SH_DIR} sw
	@echo "Running SoftHier simulation for GEMM ..."
	pld=${CUR_DIR}/software/SummaGEMM/preload.elf cfg=${ARCH_FILE} app=${CUR_DIR}/software/SummaGEMM make -C ${SH_DIR} rund
gemm-swv:gemm-pld
	@echo "Compiling SoftHier software for GEMM ..."
	cfg=${ARCH_FILE} app=${CUR_DIR}/software/SummaGEMM make -C ${SH_DIR} sw
	@echo "Running SoftHier simulation for GEMM ..."
	pld=${CUR_DIR}/software/SummaGEMM/preload.elf cfg=${ARCH_FILE} app=${CUR_DIR}/software/SummaGEMM make -C ${SH_DIR} runv pfto

# Old preload-based workflow
gemm-pre: gemm-clean gemm-cfg gemm-pld gemm-hs

gemm-run: gemm-pre
	@echo "SoftHier simulation for GEMM (preload-based) ..."
	pld=${CUR_DIR}/software/SummaGEMM/preload.elf make -C ${SH_DIR} run
gemm-runv: gemm-pre
	@echo "SoftHier simulation for GEMM (preload-based) ..."
	pld=${CUR_DIR}/software/SummaGEMM/preload.elf make -C ${SH_DIR} runv pfto
gemm-run%: gemm-pre
	@echo "SoftHier simulation for GEMM (preload-based) ..."
	pld=${CUR_DIR}/software/SummaGEMM/preload.elf make -C ${SH_DIR} run$*

# New linker script-based workflow
gemm-ls-pre: gemm-clean gemm-cfg gemm-data gemm-hs

gemm-ls-run: gemm-ls-pre
	@echo "SoftHier simulation for GEMM (linker script-based) ..."
	make -C ${SH_DIR} run
gemm-ls-runv: gemm-ls-pre
	@echo "SoftHier simulation for GEMM (linker script-based) ..."
	make -C ${SH_DIR} runv pfto
gemm-ls-run%: gemm-ls-pre
	@echo "SoftHier simulation for GEMM (linker script-based) ..."
	make -C ${SH_DIR} run$*

gemm-num:
	@echo "Numerical check for GEMM ..."
	python3 ${CUR_DIR}/scripts/kernels/gemm_numerical_check.py ${SH_DIR}/dump_0 ${GEMM_FILE}

gemm-clean:
	@echo "clean up intermidiate files generated for GEMM ..."
	rm -f ${CUR_DIR}/software/SummaGEMM/include/gemm.h \
	      ${CUR_DIR}/software/SummaGEMM/preload.elf \
	      ${CUR_DIR}/software/SummaGEMM/preload.dump \
	      ${CUR_DIR}/software/SummaGEMM/include/gemm_data.h \
	      ${CUR_DIR}/software/SummaGEMM/include/gemm_addresses.h \
	      ${SH_DIR}/dump_*
gemv-cfg:
	@echo "Generating configuration files for GEMV ..."
	python3 ${CUR_DIR}/scripts/kernels/gemm_config.py ${GEMV_FILE} ${CUR_DIR}/software/SummaGEMM/include/gemm.h

gemv-pld:
	@echo "Generating preload files for GEMV ..."
	python3 ${CUR_DIR}/scripts/kernels/gemm_preload.py \
			${CUR_DIR}/software/SummaGEMM/preload.elf \
			${CUR_DIR}/software/SummaGEMM/include/gemm_addresses.h \
			${ARCH_FILE} \
			${GEMV_FILE}

gemv-data:
	@echo "Generating data header for GEMV (linker script-based) ..."
	python3 ${CUR_DIR}/scripts/kernels/gemm_data_generator.py \
			${CUR_DIR}/software/SummaGEMM/include/gemm_data.h \
			${ARCH_FILE} \
			${GEMV_FILE}

gemv-hs:
	@echo "Compiling SoftHier HW & software for GEMV ..."
	cfg=${ARCH_FILE} app=${CUR_DIR}/software/SummaGEMM make -C ../../../../ hs

gemv-sw: gemv-cfg
	@echo "Compiling SoftHier software for GEMV ..."
	cfg=${ARCH_FILE} app=${CUR_DIR}/software/SummaGEMM make -C ${SH_DIR} sw
	@echo "Running SoftHier simulation for GEMV ..."
	pld=${CUR_DIR}/software/SummaGEMM/preload.elf cfg=${ARCH_FILE} app=${CUR_DIR}/software/SummaGEMM make -C ${SH_DIR} run
gemv-swd:gemv-pld
	@echo "Compiling SoftHier software for GEMV ..."
	cfg=${ARCH_FILE} app=${CUR_DIR}/software/SummaGEMM make -C ${SH_DIR} sw
	@echo "Running SoftHier simulation for GEMV ..."
	pld=${CUR_DIR}/software/SummaGEMM/preload.elf cfg=${ARCH_FILE} app=${CUR_DIR}/software/SummaGEMM make -C ${SH_DIR} rund
gemv-swv:gemv-pld
	@echo "Compiling SoftHier software for GEMV ..."
	cfg=${ARCH_FILE} app=${CUR_DIR}/software/SummaGEMM make -C ${SH_DIR} sw
	@echo "Running SoftHier simulation for GEMV ..."
	pld=${CUR_DIR}/software/SummaGEMM/preload.elf cfg=${ARCH_FILE} app=${CUR_DIR}/software/SummaGEMM make -C ${SH_DIR} runv pfto

# Old preload-based workflow
gemv-pre: gemv-clean gemv-cfg gemv-pld gemv-hs

gemv-run: gemv-pre
	@echo "SoftHier simulation for GEMV (preload-based) ..."
	pld=${CUR_DIR}/software/SummaGEMM/preload.elf make -C ${SH_DIR} run
gemv-runv: gemv-pre
	@echo "SoftHier simulation for GEMV (preload-based) ..."
	pld=${CUR_DIR}/software/SummaGEMM/preload.elf make -C ${SH_DIR} runv pfto
gemv-run%: gemv-pre
	@echo "SoftHier simulation for GEMV (preload-based) ..."
	pld=${CUR_DIR}/software/SummaGEMM/preload.elf make -C ${SH_DIR} run$*

# New linker script-based workflow
gemv-ls-pre: gemv-clean gemv-cfg gemv-data gemv-hs

gemv-ls-run: gemv-ls-pre
	@echo "SoftHier simulation for GEMV (linker script-based) ..."
	make -C ${SH_DIR} run
gemv-ls-runv: gemv-ls-pre
	@echo "SoftHier simulation for GEMV (linker script-based) ..."
	make -C ${SH_DIR} runv pfto
gemv-ls-run%: gemv-ls-pre
	@echo "SoftHier simulation for GEMV (linker script-based) ..."
	make -C ${SH_DIR} run$*

gemv-num:
	@echo "Numerical check for GEMV ..."
	python3 ${CUR_DIR}/scripts/kernels/gemm_numerical_check.py ${SH_DIR}/dump_0 ${GEMV_FILE}

gemv-clean:
	@echo "clean up intermidiate files generated for GEMV ..."
	rm -f ${CUR_DIR}/software/SummaGEMM/include/gemm.h \
	      ${CUR_DIR}/software/SummaGEMM/preload.elf \
	      ${CUR_DIR}/software/SummaGEMM/preload.dump \
	      ${CUR_DIR}/software/SummaGEMM/include/gemm_data.h \
	      ${CUR_DIR}/software/SummaGEMM/include/gemm_addresses.h \
	      ${SH_DIR}/dump_*
# ============================================================================
# VQ (Vector Quantization) Workflow Targets
# ============================================================================

# VQ cache directory
VQ_CACHE_DIR ?= ${CUR_DIR}/vq_cache

# VQ data generation from pretrained model
# Usage: make vq-download REPO_ID=ISTA-DASLab/Llama-2-7b-AQLM-PV-2Bit-2x8-hf LAYER=model.layers.0.mlp.down_proj
REPO_ID ?= ISTA-DASLab/Llama-2-7b-AQLM-2Bit-2x8-hf
WEIGHT_FILE ?= model.safetensors
LAYER_PREFIX ?= model.layers.0.mlp.down_proj
VQ_K ?= 2048
VQ_N ?= 2048
vq-download:
	@echo "Downloading and extracting VQ model from HuggingFace..."
	@mkdir -p ${VQ_CACHE_DIR}
	python3 ${CUR_DIR}/scripts/quantizers/download.py \
		--repo_id ${REPO_ID} \
		--weight_file ${WEIGHT_FILE} \
		--layer_prefix ${LAYER_PREFIX} \
		--output_dir ${VQ_CACHE_DIR} \
		--k_size ${VQ_K} \
		--n_size ${VQ_N}

# Generate VQ data using AQLM quantizer (for custom weights)
# This generates random weights and quantizes them using AQLM
VQ_NUM_CB ?= 2
VQ_NBITS_PER_CB ?= 8
VQ_GROUP_SIZE ?= 8
VQ_MAX_EPOCHS ?= 20
VQ_DEVICE ?= auto

vq-generate:
	@echo "Generating and quantizing weights using AQLM..."
	@mkdir -p ${VQ_CACHE_DIR}
	python3 ${CUR_DIR}/scripts/quantizers/aqlm_quantizer.py \
		--k_size ${VQ_K} \
		--n_size ${VQ_N} \
		--num_codebooks ${VQ_NUM_CB} \
		--nbits_per_codebook ${VQ_NBITS_PER_CB} \
		--in_group_size ${VQ_GROUP_SIZE} \
		--max_epochs ${VQ_MAX_EPOCHS} \
		--device ${VQ_DEVICE} \
		--output_dir ${VQ_CACHE_DIR}

# Check VQ cache status
vq-status:
	@echo "VQ Cache Directory: ${VQ_CACHE_DIR}"
	@if [ -d "${VQ_CACHE_DIR}" ]; then \
		echo "Cache contents:"; \
		ls -lh ${VQ_CACHE_DIR}/*.npy 2>/dev/null || echo "  No .npy files found"; \
	else \
		echo "Cache directory does not exist"; \
	fi

# Clean VQ cache
vq-clean:
	@echo "Cleaning VQ cache..."
	rm -rf ${VQ_CACHE_DIR}

# GEMM with VQ - Full workflow
gemm-vq-pre: vq-status gemm-clean gemm-cfg gemm-pld gemm-hs

gemm-vq-run: gemm-vq-pre
	@echo "Running GEMM with VQ..."
	pld=${CUR_DIR}/software/SummaGEMM/preload.elf make -C ${SH_DIR} run

gemm-vq-runv: gemm-vq-pre
	@echo "Running GEMM with VQ (verbose)..."
	pld=${CUR_DIR}/software/SummaGEMM/preload.elf make -C ${SH_DIR} runv pfto

# Help target for VQ workflow
vq-help:
	@echo "Vector Quantization (VQ) Workflow Targets:"
	@echo ""
	@echo "  vq-download      Download pretrained VQ model from HuggingFace"
	@echo "                   Variables: REPO_ID, WEIGHT_FILE, LAYER_PREFIX, VQ_K, VQ_N"
	@echo "                   Saves to: aqlm_dl{config}_dim{K}x{N}_*.npy"
	@echo ""
	@echo "  vq-generate      Generate and quantize random weights using AQLM"
	@echo "                   Variables: VQ_K, VQ_N, VQ_NUM_CB, VQ_NBITS_PER_CB, VQ_GROUP_SIZE, VQ_MAX_EPOCHS, VQ_DEVICE"
	@echo "                   Saves to: aqlm_gen{config}_dim{K}x{N}_*.npy"
	@echo ""
	@echo "  vq-status        Show VQ cache status"
	@echo "  vq-clean         Clean VQ cache directory"
	@echo ""
	@echo "  gemm-vq-pre      Prepare GEMM with VQ (config + preload + compile)"
	@echo "  gemm-vq-run      Run GEMM simulation with VQ"
	@echo "  gemm-vq-runv     Run GEMM with VQ (verbose, with waveforms)"
	@echo ""
	@echo "VQ Source Selection (config/kernels/gemm.py):"
	@echo "  self.vq_source = \"gen\"    Use generated weights (aqlm_gen* files)"
	@echo "  self.vq_source = \"dl\"     Use downloaded weights (aqlm_dl* files)"
	@echo ""
	@echo "Example workflows:"
	@echo "  1. Generate VQ data and run:"
	@echo "     make vq-generate"
	@echo "     # Set vq_source='gen' and vq_force_weight_load=1 in gemm.py"
	@echo "     make gemm-vq-run"
	@echo ""
	@echo "  2. Download pretrained model and run:"
	@echo "     make vq-download REPO_ID=... LAYER_PREFIX=..."
	@echo "     # Set vq_source='dl' and vq_force_weight_load=1 in gemm.py"
	@echo "     make gemm-vq-run"
	@echo ""
	@echo "  3. Keep both and switch between them:"
	@echo "     make vq-generate    # Creates aqlm_gen* files"
	@echo "     make vq-download    # Creates aqlm_dl* files"
	@echo "     # Toggle vq_source='gen' or 'dl' in gemm.py to switch"
	@echo ""
	@echo "Config Flags (config/kernels/gemm.py):"
	@echo "  vq_enabled           = 0/1    # Enable VQ hardware support"
	@echo "  vq_force_weight_load = 0/1    # Load VQ weights even if HW disabled"
	@echo "  vq_source            = gen/dl # Select generated or downloaded weights"

# ============================================================================
# VQ Debugging Targets
# ============================================================================

# Quick debug commands using the debug toolkit
vq-debug-summary:
	@echo "VQ Data Summary:"
	./scripts/quick_debug.sh summary

vq-debug-cb:
	@echo "Inspecting Codebooks:"
	./scripts/quick_debug.sh cb

vq-debug-idx:
	@echo "Inspecting Indices:"
	./scripts/quick_debug.sh idx

vq-debug-compare:
	@echo "Comparing W_orig vs W_hat:"
	./scripts/quick_debug.sh compare

# Manual dequantization (specify ROW and GROUP)
# Usage: make vq-debug-dq ROW=0 GROUP=0
ROW ?= 0
GROUP ?= 0
vq-debug-dq:
	@echo "Manual dequantization for Row $(ROW), Group $(GROUP):"
	./scripts/quick_debug.sh dq $(ROW) $(GROUP)

# Full debug workflow - run this to verify VQ data after generation
vq-debug-full:
	@echo "=== VQ Full Debug Workflow ==="
	@echo ""
	@echo "1. Summary:"
	@./scripts/quick_debug.sh summary
	@echo ""
	@echo "2. Codebook 0 (first 5 centroids):"
	@./scripts/quick_debug.sh cb 0
	@echo ""
	@echo "3. Indices Row 0 (first 10 groups):"
	@./scripts/quick_debug.sh idx 0
	@echo ""
	@echo "4. Scales (first 10 rows):"
	@./scripts/quick_debug.sh scales 10
	@echo ""
	@echo "5. Manual dequantization (row 0, group 0):"
	@./scripts/quick_debug.sh dq 0 0
	@echo ""
	@echo "6. Comparison W_orig vs W_hat:"
	@./scripts/quick_debug.sh compare
	@echo ""
	@echo "=== Debug Complete ==="

# Help for debugging targets
vq-debug-help:
	@echo "VQ Debugging Targets:"
	@echo ""
	@echo "  vq-debug-summary     Show VQ data summary"
	@echo "  vq-debug-cb          Inspect codebooks"
	@echo "  vq-debug-idx         Inspect indices"
	@echo "  vq-debug-compare     Compare W_orig vs W_hat"
	@echo "  vq-debug-dq          Manual dequantization (set ROW= and GROUP=)"
	@echo "  vq-debug-full        Run complete debug workflow"
	@echo ""
	@echo "Examples:"
	@echo "  make vq-debug-summary"
	@echo "  make vq-debug-dq ROW=5 GROUP=10"
	@echo "  make vq-debug-full"
	@echo ""
	@echo "For more options, run: ./scripts/quick_debug.sh help"
	@echo "Full guide: cat DEBUG_GUIDE.md"


# Clang-tidy targets
tidy:
	clang-tidy-18 software/*.c -- \
		-target riscv32-unknown-elf \
		-isystem $(abspath ../../../../third_party/toolchain/install/lib/gcc/riscv32-unknown-elf/14.2.0/include) \
		-isystem $(abspath ../../../../third_party/toolchain/install/riscv32-unknown-elf/include) \
		-I$(abspath ../../../../soft_hier/flex_cluster_sdk/runtime/include) \
		-I$(abspath software/include) \
		-D__riscv -D__riscv_xlen=32


