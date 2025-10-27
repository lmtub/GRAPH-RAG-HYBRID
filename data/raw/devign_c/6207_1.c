static void gen_mtsr(DisasContext *ctx)

{

#if defined(CONFIG_USER_ONLY)

    gen_inval_exception(ctx, POWERPC_EXCP_PRIV_REG);

#else

    TCGv t0;

    if (unlikely(ctx->pr)) {

        gen_inval_exception(ctx, POWERPC_EXCP_PRIV_REG);

        return;

    }

    t0 = tcg_const_tl(SR(ctx->opcode));

    gen_helper_store_sr(cpu_env, t0, cpu_gpr[rS(ctx->opcode)]);

    tcg_temp_free(t0);

#endif

}
