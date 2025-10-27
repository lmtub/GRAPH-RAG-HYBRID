static void gen_mtfsfi(DisasContext *ctx)

{

    int bf, sh;

    TCGv_i64 t0;

    TCGv_i32 t1;



    if (unlikely(!ctx->fpu_enabled)) {

        gen_exception(ctx, POWERPC_EXCP_FPU);

        return;

    }

    bf = crbD(ctx->opcode) >> 2;

    sh = 7 - bf;

    /* NIP cannot be restored if the memory exception comes from an helper */

    gen_update_nip(ctx, ctx->nip - 4);

    gen_reset_fpstatus();

    t0 = tcg_const_i64(FPIMM(ctx->opcode) << (4 * sh));

    t1 = tcg_const_i32(1 << sh);

    gen_helper_store_fpscr(cpu_env, t0, t1);

    tcg_temp_free_i64(t0);

    tcg_temp_free_i32(t1);

    if (unlikely(Rc(ctx->opcode) != 0)) {

        tcg_gen_trunc_tl_i32(cpu_crf[1], cpu_fpscr);

        tcg_gen_shri_i32(cpu_crf[1], cpu_crf[1], FPSCR_OX);

    }

    /* We can raise a differed exception */

    gen_helper_float_check_status(cpu_env);

}
