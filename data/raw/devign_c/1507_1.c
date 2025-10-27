static inline void gen_op_arith_add(DisasContext *ctx, TCGv ret, TCGv arg1,

                                    TCGv arg2, int add_ca, int compute_ca,

                                    int compute_ov)

{

    TCGv t0, t1;



    if ((!compute_ca && !compute_ov) ||

        (!TCGV_EQUAL(ret,arg1) && !TCGV_EQUAL(ret, arg2)))  {

        t0 = ret;

    } else {

        t0 = tcg_temp_local_new();

    }



    if (add_ca) {

        t1 = tcg_temp_local_new();

        tcg_gen_mov_tl(t1, cpu_ca);

    } else {

        TCGV_UNUSED(t1);

    }



    if (compute_ca) {

        /* Start with XER CA disabled, the most likely case */

        tcg_gen_movi_tl(cpu_ca, 0);

    }

    if (compute_ov) {

        /* Start with XER OV disabled, the most likely case */

        tcg_gen_movi_tl(cpu_ov, 0);

    }



    tcg_gen_add_tl(t0, arg1, arg2);



    if (compute_ca) {

        gen_op_arith_compute_ca(ctx, t0, arg1, 0);

    }

    if (add_ca) {

        tcg_gen_add_tl(t0, t0, t1);

        gen_op_arith_compute_ca(ctx, t0, t1, 0);

        tcg_temp_free(t1);

    }

    if (compute_ov) {

        gen_op_arith_compute_ov(ctx, t0, arg1, arg2, 0);

    }



    if (unlikely(Rc(ctx->opcode) != 0))

        gen_set_Rc0(ctx, t0);



    if (!TCGV_EQUAL(t0, ret)) {

        tcg_gen_mov_tl(ret, t0);

        tcg_temp_free(t0);

    }

}
