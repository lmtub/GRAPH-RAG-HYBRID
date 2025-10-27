static void gen_mtsrin_64b(DisasContext *ctx)

{

#if defined(CONFIG_USER_ONLY)

    gen_inval_exception(ctx, POWERPC_EXCP_PRIV_REG);

#else

    TCGv t0;

    if (unlikely(ctx->pr)) {

        gen_inval_exception(ctx, POWERPC_EXCP_PRIV_REG);

        return;

    }

    t0 = tcg_temp_new();

    tcg_gen_shri_tl(t0, cpu_gpr[rB(ctx->opcode)], 28);

    tcg_gen_andi_tl(t0, t0, 0xF);

    gen_helper_store_sr(cpu_env, t0, cpu_gpr[rS(ctx->opcode)]);

    tcg_temp_free(t0);

#endif

}
