static void gen_mtfsf(DisasContext *ctx)

{

    TCGv_i32 t0;

    int L = ctx->opcode & 0x02000000;



    if (unlikely(!ctx->fpu_enabled)) {

        gen_exception(ctx, POWERPC_EXCP_FPU);

        return;

    }

    /* NIP cannot be restored if the memory exception comes from an helper */

    gen_update_nip(ctx, ctx->nip - 4);

    gen_reset_fpstatus();

    if (L)

        t0 = tcg_const_i32(0xff);

    else

        t0 = tcg_const_i32(FM(ctx->opcode));

    gen_helper_store_fpscr(cpu_env, cpu_fpr[rB(ctx->opcode)], t0);

    tcg_temp_free_i32(t0);

    if (unlikely(Rc(ctx->opcode) != 0)) {

        tcg_gen_trunc_tl_i32(cpu_crf[1], cpu_fpscr);

        tcg_gen_shri_i32(cpu_crf[1], cpu_crf[1], FPSCR_OX);

    }

    /* We can raise a differed exception */

    gen_helper_float_check_status(cpu_env);

}
