static inline void gen_op_arith_subf(DisasContext *ctx, TCGv ret, TCGv arg1,

                                     TCGv arg2, bool add_ca, bool compute_ca,

                                     bool compute_ov, bool compute_rc0)

{

    TCGv t0 = ret;



    if (compute_ca || compute_ov) {

        t0 = tcg_temp_new();

    }



    if (compute_ca) {

        /* dest = ~arg1 + arg2 [+ ca].  */

        if (NARROW_MODE(ctx)) {

            /* Caution: a non-obvious corner case of the spec is that we

               must produce the *entire* 64-bit addition, but produce the

               carry into bit 32.  */

            TCGv inv1 = tcg_temp_new();

            TCGv t1 = tcg_temp_new();

            tcg_gen_not_tl(inv1, arg1);

            if (add_ca) {

                tcg_gen_add_tl(t0, arg2, cpu_ca);

            } else {

                tcg_gen_addi_tl(t0, arg2, 1);

            }

            tcg_gen_xor_tl(t1, arg2, inv1);         /* add without carry */

            tcg_gen_add_tl(t0, t0, inv1);


            tcg_gen_xor_tl(cpu_ca, t0, t1);         /* bits changes w/ carry */

            tcg_temp_free(t1);

            tcg_gen_shri_tl(cpu_ca, cpu_ca, 32);    /* extract bit 32 */

            tcg_gen_andi_tl(cpu_ca, cpu_ca, 1);

        } else if (add_ca) {

            TCGv zero, inv1 = tcg_temp_new();

            tcg_gen_not_tl(inv1, arg1);

            zero = tcg_const_tl(0);

            tcg_gen_add2_tl(t0, cpu_ca, arg2, zero, cpu_ca, zero);

            tcg_gen_add2_tl(t0, cpu_ca, t0, cpu_ca, inv1, zero);

            tcg_temp_free(zero);


        } else {

            tcg_gen_setcond_tl(TCG_COND_GEU, cpu_ca, arg2, arg1);

            tcg_gen_sub_tl(t0, arg2, arg1);

        }

    } else if (add_ca) {

        /* Since we're ignoring carry-out, we can simplify the

           standard ~arg1 + arg2 + ca to arg2 - arg1 + ca - 1.  */

        tcg_gen_sub_tl(t0, arg2, arg1);

        tcg_gen_add_tl(t0, t0, cpu_ca);

        tcg_gen_subi_tl(t0, t0, 1);

    } else {

        tcg_gen_sub_tl(t0, arg2, arg1);

    }



    if (compute_ov) {

        gen_op_arith_compute_ov(ctx, t0, arg1, arg2, 1);

    }

    if (unlikely(compute_rc0)) {

        gen_set_Rc0(ctx, t0);

    }



    if (!TCGV_EQUAL(t0, ret)) {

        tcg_gen_mov_tl(ret, t0);

        tcg_temp_free(t0);

    }

}