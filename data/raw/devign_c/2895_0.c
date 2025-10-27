static inline void gen_op_mov_v_reg(int ot, TCGv t0, int reg)

{

    switch(ot) {

    case OT_BYTE:

        if (reg < 4 X86_64_DEF( || reg >= 8 || x86_64_hregs)) {

            goto std_case;

        } else {

            tcg_gen_shri_tl(t0, cpu_regs[reg - 4], 8);

            tcg_gen_ext8u_tl(t0, t0);

        }

        break;

    default:

    std_case:

        tcg_gen_mov_tl(t0, cpu_regs[reg]);

        break;

    }

}
