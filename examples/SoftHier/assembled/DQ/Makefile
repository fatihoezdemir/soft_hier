pld_gemm:
	cd software/util; python store_cb_and_idx.py -kernel gemm
pld_gemv:
	cd software/util; python store_cb_and_idx.py -kernel gemv
dq: 
	DEBUG=1 cfg=$(abspath config/arch_dq_gemv_sc.py) app=$(abspath software) make -C ../../../../ hs run
dqsw:
	DEBUG=1 cfg=$(abspath config/arch_dq_gemv_sc.py) app=$(abspath software) make -C ../../../../ sw run
dq_nodebug: 
	cfg=$(abspath config/arch_dq_gemv_sc.py) app=$(abspath software)  make -C ../../../../ hs run
pfto: 
	DEBUG=1 cfg=$(abspath config/arch_dq_gemv_sc.py) app=$(abspath software) make -C ../../../../ hs runv pfto