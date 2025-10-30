pld_gemm:
	cd software/util; python store_cb_and_idx.py -kernel gemm
pld_gemv:
	cd software/util; python store_cb_and_idx.py -kernel gemv
dq:
	DEBUG=1 cfg=$(abspath config/arch_dq_sc.py) app=$(abspath software) make -C ../../../../ hs run
dqsw:
	DEBUG=1 cfg=$(abspath config/arch_dq_sc.py) app=$(abspath software) make -C ../../../../ sw run 
dq_nodebug:
	cfg=$(abspath config/arch_dq_sc.py) app=$(abspath software)  make -C ../../../../ hs run
pfto:
	 VERIFY_VALUES=0 DEBUG=1 cfg=$(abspath config/arch_dq_sc.py) app=$(abspath software) make -C ../../../../ hs runv pfto
rund:
	DEBUG=1 cfg=$(abspath config/arch_dq_sc.py) app=$(abspath software) make -C ../../../../ hs rund 
dqswv:
	make pld_gemv;
	DEBUG=1 cfg=$(abspath config/arch_dq_sc.py) app=$(abspath software) make -C ../../../../ sw run
dqswm:
	make pld_gemm;
	DEBUG=1 cfg=$(abspath config/arch_dq_sc.py) app=$(abspath software) make -C ../../../../ sw run

# Test targets
test:
	DEBUG=1 cfg=$(abspath config/arch_dq_sc.py) app=$(abspath software) make -C ../../../../ hs
	./../../../../install/bin/gvsoc --target=pulp.chips.flex_cluster.flex_cluster --binary ../../../../sw_build/softhier_test.elf run

testsw:
	DEBUG=1 cfg=$(abspath config/arch_dq_sc.py) app=$(abspath software) make -C ../../../../ sw
	./../../../../install/bin/gvsoc --target=pulp.chips.flex_cluster.flex_cluster --binary ../../../../sw_build/softhier_test.elf run

# Clang-tidy targets
tidy:
	clang-tidy-18 software/*.c -- \
		-target riscv32-unknown-elf \
		-isystem $(abspath ../../../../third_party/toolchain/install/lib/gcc/riscv32-unknown-elf/14.2.0/include) \
		-isystem $(abspath ../../../../third_party/toolchain/install/riscv32-unknown-elf/include) \
		-I$(abspath ../../../../soft_hier/flex_cluster_sdk/runtime/include) \
		-I$(abspath software/include) \
		-D__riscv -D__riscv_xlen=32


