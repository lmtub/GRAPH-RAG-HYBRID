static void gen_compute_eflags_s(DisasContext *s, TCGv reg, bool inv)

{

    switch (s->cc_op) {

    case CC_OP_DYNAMIC:

        gen_compute_eflags(s);

        /* FALLTHRU */

    case CC_OP_EFLAGS:

        tcg_gen_shri_tl(reg, cpu_cc_src, 7);

        tcg_gen_andi_tl(reg, reg, 1);

        if (inv) {

            tcg_gen_xori_tl(reg, reg, 1);

        }

        break;

    default:

        {

            int size = (s->cc_op - CC_OP_ADDB) & 3;

            TCGv t0 = gen_ext_tl(reg, cpu_cc_dst, size, true);

            tcg_gen_setcondi_tl(inv ? TCG_COND_GE : TCG_COND_LT, reg, t0, 0);

        }

        break;

    }

}
