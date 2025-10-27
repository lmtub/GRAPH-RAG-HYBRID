static inline void gen_op_eval_bge(TCGv dst, TCGv_i32 src)

{

    gen_mov_reg_V(cpu_tmp0, src);

    gen_mov_reg_N(dst, src);

    tcg_gen_xor_tl(dst, dst, cpu_tmp0);

    tcg_gen_xori_tl(dst, dst, 0x1);

}
