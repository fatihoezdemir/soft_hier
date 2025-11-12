MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
CUR_DIR := $(MAKEFILE_DIR)
SH_DIR := $(realpath $(MAKEFILE_DIR)../../../..)

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
GEMM_FILE ?= $(CUR_DIR)config/kernels/gemm.py
ifdef gemm
	GEMM_FILE = $(realpath $(gemm))
endif
ARCH_FILE = ${CUR_DIR}/config/arch/arch_summa.py
gemm-cfg:
	@echo "Generating configuration files for GEMM ..."
	python3 ${CUR_DIR}/scripts/kernels/gemm_config.py ${GEMM_FILE} ${CUR_DIR}/software/SummaGEMM/include/gemm.h

gemm-pld:
	@echo "Generating preload files for GEMM ..."
	python3 ${CUR_DIR}/scripts/kernels/gemm_preload.py \
			${CUR_DIR}/software/SummaGEMM/preload.elf \
			${CUR_DIR}/software/SummaGEMM/include/preload.h \
			${ARCH_FILE} \
			${GEMM_FILE}
gemm-hs:
	@echo "Compiling SoftHier HW & software for GEMM ..."
	cfg=${ARCH_FILE} app=${CUR_DIR}/software/SummaGEMM make -C ../../../../ hs

gemm-sw:
	@echo "Compiling SoftHier HW & software for GEMM ..."
	cfg=${ARCH_FILE} app=${CUR_DIR}/software/SummaGEMM make -C ${SH_DIR} software

gemm-pre: gemm-clean gemm-cfg gemm-pld gemm-hs

gemm-run: gemm-pre
	@echo "SoftHier simulation for GEMM ..."
	pld=${CUR_DIR}/software/SummaGEMM/preload.elf make -C ${SH_DIR} run
gemm-runv: gemm-pre
	@echo "SoftHier simulation for GEMM ..."
	pld=${CUR_DIR}/software/SummaGEMM/preload.elf make -C ${SH_DIR} runv pfto
gemm-run%: gemm-pre
	@echo "SoftHier simulation for GEMM ..."
	pld=${CUR_DIR}/software/SummaGEMM/preload.elf make -C ${SH_DIR} run$*

gemm-num:
	@echo "Numerical check for GEMM ..."
	python3 ${CUR_DIR}/scripts/kernels/gemm_numerical_check.py ${SH_DIR}/dump_0 ${GEMM_FILE}

gemm-clean:
	@echo "clean up intermidiate files generated for GEMM ..."
	rm -f ${CUR_DIR}/software/SummaGEMM/include/gemm.h ${CUR_DIR}/software/SummaGEMM/preload.elf ${CUR_DIR}/software/SummaGEMM/preload.dump ${CUR_DIR}/software/SummaGEMM/include/preload.h ${SH_DIR}/dump_*

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
VQ_K ?= 512
VQ_N ?= 512

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
# This requires you to have already trained a quantizer or have weights to quantize
vq-quantize:
	@echo "Quantizing weights using AQLM..."
	@mkdir -p ${VQ_CACHE_DIR}
	python3 ${CUR_DIR}/scripts/quantizers/aqlm_quantize.py \
		--output_dir ${VQ_CACHE_DIR} \
		--m_size ${VQ_M} \
		--n_size ${VQ_N}

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
	@echo "                   Variables: REPO_ID, WEIGHT_FILE, LAYER_PREFIX, VQ_M, VQ_N"
	@echo "  vq-quantize      Quantize custom weights using AQLM"
	@echo "  vq-status        Show VQ cache status"
	@echo "  vq-clean         Clean VQ cache directory"
	@echo ""
	@echo "  gemm-vq-pre      Prepare GEMM with VQ (config + preload + compile)"
	@echo "  gemm-vq-run      Run GEMM simulation with VQ"
	@echo "  gemm-vq-runv     Run GEMM with VQ (verbose, with waveforms)"
	@echo ""
	@echo "Example workflows:"
	@echo "  1. Using cached VQ data:"
	@echo "     make vq-status && make gemm-vq-run"
	@echo ""
	@echo "  2. Download pretrained model and run:"
	@echo "     make vq-download REPO_ID=... LAYER_PREFIX=... && make gemm-vq-run"
	@echo ""
	@echo "Note: VQ is controlled by vq_enabled flag in config/kernels/gemm.py"



# Clang-tidy targets
tidy:
	clang-tidy-18 software/*.c -- \
		-target riscv32-unknown-elf \
		-isystem $(abspath ../../../../third_party/toolchain/install/lib/gcc/riscv32-unknown-elf/14.2.0/include) \
		-isystem $(abspath ../../../../third_party/toolchain/install/riscv32-unknown-elf/include) \
		-I$(abspath ../../../../soft_hier/flex_cluster_sdk/runtime/include) \
		-I$(abspath software/include) \
		-D__riscv -D__riscv_xlen=32


