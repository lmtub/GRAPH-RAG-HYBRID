static void gen_mfsri(DisasContext *ctx)

{

#if defined(CONFIG_USER_ONLY)

    gen_inval_exception(ctx, POWERPC_EXCP_PRIV_OPC);

#else

    int ra = rA(ctx->opcode);

    int rd = rD(ctx->opcode);

    TCGv t0;

    if (unlikely(ctx->pr)) {

        gen_inval_exception(ctx, POWERPC_EXCP_PRIV_OPC);

        return;

    }

    t0 = tcg_temp_new();

    gen_addr_reg_index(ctx, t0);

    tcg_gen_shri_tl(t0, t0, 28);

    tcg_gen_andi_tl(t0, t0, 0xF);

    gen_helper_load_sr(cpu_gpr[rd], cpu_env, t0);

    tcg_temp_free(t0);

    if (ra != 0 && ra != rd)

        tcg_gen_mov_tl(cpu_gpr[ra], cpu_gpr[rd]);

#endif

}
