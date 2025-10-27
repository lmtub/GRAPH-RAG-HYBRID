static void gen_lswi(DisasContext *ctx)

{

    TCGv t0;

    TCGv_i32 t1, t2;

    int nb = NB(ctx->opcode);

    int start = rD(ctx->opcode);

    int ra = rA(ctx->opcode);

    int nr;



    if (nb == 0)

        nb = 32;

    nr = (nb + 3) / 4;

    if (unlikely(lsw_reg_in_range(start, nr, ra))) {

        gen_inval_exception(ctx, POWERPC_EXCP_INVAL_LSWX);

        return;

    }

    gen_set_access_type(ctx, ACCESS_INT);

    /* NIP cannot be restored if the memory exception comes from an helper */

    gen_update_nip(ctx, ctx->nip - 4);

    t0 = tcg_temp_new();

    gen_addr_register(ctx, t0);

    t1 = tcg_const_i32(nb);

    t2 = tcg_const_i32(start);

    gen_helper_lsw(cpu_env, t0, t1, t2);

    tcg_temp_free(t0);

    tcg_temp_free_i32(t1);

    tcg_temp_free_i32(t2);

}
