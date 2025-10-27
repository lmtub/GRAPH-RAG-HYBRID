static inline void gen_op_eval_fblg(TCGv dst, TCGv src,

                                    unsigned int fcc_offset)

{

    gen_mov_reg_FCC0(dst, src, fcc_offset);

    gen_mov_reg_FCC1(cpu_tmp0, src, fcc_offset);

    tcg_gen_xor_tl(dst, dst, cpu_tmp0);

}
