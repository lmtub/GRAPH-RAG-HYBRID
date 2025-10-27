static void gen_lswx(DisasContext *ctx)

{

    TCGv t0;

    TCGv_i32 t1, t2, t3;

    gen_set_access_type(ctx, ACCESS_INT);

    /* NIP cannot be restored if the memory exception comes from an helper */

    gen_update_nip(ctx, ctx->nip - 4);

    t0 = tcg_temp_new();

    gen_addr_reg_index(ctx, t0);

    t1 = tcg_const_i32(rD(ctx->opcode));

    t2 = tcg_const_i32(rA(ctx->opcode));

    t3 = tcg_const_i32(rB(ctx->opcode));

    gen_helper_lswx(cpu_env, t0, t1, t2, t3);

    tcg_temp_free(t0);

    tcg_temp_free_i32(t1);

    tcg_temp_free_i32(t2);

    tcg_temp_free_i32(t3);

}
