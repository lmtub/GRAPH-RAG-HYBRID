static inline void gen_op_movl_seg_T0_vm(int seg_reg)

{

    tcg_gen_andi_tl(cpu_T[0], cpu_T[0], 0xffff);

    tcg_gen_st32_tl(cpu_T[0], cpu_env, 

                    offsetof(CPUX86State,segs[seg_reg].selector));

    tcg_gen_shli_tl(cpu_T[0], cpu_T[0], 4);

    tcg_gen_st_tl(cpu_T[0], cpu_env, 

                  offsetof(CPUX86State,segs[seg_reg].base));

}
