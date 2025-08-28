pld:
	cd preload; python preload_matrices.py

dq_gemv_sc: pld
	cfg=$(abspath config/arch_dq_gemv_sc.py) app=$(abspath software) pld=$(abspath preload/myy_preload.elf) make -C ../../../../ hs run
