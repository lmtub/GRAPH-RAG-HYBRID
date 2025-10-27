static inline void gen_op_jz_ecx(TCGMemOp size, int label1)

{

    tcg_gen_mov_tl(cpu_tmp0, cpu_regs[R_ECX]);

    gen_extu(size, cpu_tmp0);

    tcg_gen_brcondi_tl(TCG_COND_EQ, cpu_tmp0, 0, label1);

}
