static inline void gen_op_eval_fbe(TCGv dst, TCGv src,

                                    unsigned int fcc_offset)

{

    gen_mov_reg_FCC0(dst, src, fcc_offset);

    gen_mov_reg_FCC1(cpu_tmp0, src, fcc_offset);

    tcg_gen_or_tl(dst, dst, cpu_tmp0);

    tcg_gen_xori_tl(dst, dst, 0x1);

}
