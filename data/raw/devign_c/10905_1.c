static void gen_mfmsr(DisasContext *ctx)

{

#if defined(CONFIG_USER_ONLY)

    gen_inval_exception(ctx, POWERPC_EXCP_PRIV_REG);

#else

    if (unlikely(ctx->pr)) {

        gen_inval_exception(ctx, POWERPC_EXCP_PRIV_REG);

        return;

    }

    tcg_gen_mov_tl(cpu_gpr[rD(ctx->opcode)], cpu_msr);

#endif

}
