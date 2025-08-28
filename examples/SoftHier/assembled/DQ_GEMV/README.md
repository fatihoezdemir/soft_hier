cfg=examples/SoftHier/assembled/DQ_GEMV/config/arch_dq_gemv_sc.py app=examples/SoftHier/assembled/DQ_GEMV/software pld=examples/myy_preload.elf make hs; make run

 app=examples/SoftHier/assembled/DQ_GEMV/software make sw; timeout 30 make run

