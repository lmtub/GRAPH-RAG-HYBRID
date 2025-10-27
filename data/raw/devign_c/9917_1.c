static always_inline void gen_op_arith_compute_ca(DisasContext *ctx, TCGv arg1, TCGv arg2, int sub)

{

    int l1 = gen_new_label();



#if defined(TARGET_PPC64)

    if (!(ctx->sf_mode)) {

        TCGv t0, t1;

        t0 = tcg_temp_new(TCG_TYPE_TL);

        t1 = tcg_temp_new(TCG_TYPE_TL);



        tcg_gen_ext32u_tl(t0, arg1);

        tcg_gen_ext32u_tl(t1, arg2);

        if (sub) {

            tcg_gen_brcond_tl(TCG_COND_GTU, t0, t1, l1);

        } else {

            tcg_gen_brcond_tl(TCG_COND_GEU, t0, t1, l1);

        }

    } else

#endif

    if (sub) {

        tcg_gen_brcond_tl(TCG_COND_GTU, arg1, arg2, l1);

    } else {

        tcg_gen_brcond_tl(TCG_COND_GEU, arg1, arg2, l1);

    }

    tcg_gen_ori_tl(cpu_xer, cpu_xer, 1 << XER_CA);

    gen_set_label(l1);

}
