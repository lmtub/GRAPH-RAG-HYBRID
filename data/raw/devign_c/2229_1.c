static void gen_mfrom(DisasContext *ctx)

{

#if defined(CONFIG_USER_ONLY)

    gen_inval_exception(ctx, POWERPC_EXCP_PRIV_OPC);

#else

    if (unlikely(ctx->pr)) {

        gen_inval_exception(ctx, POWERPC_EXCP_PRIV_OPC);

        return;

    }

    gen_helper_602_mfrom(cpu_gpr[rD(ctx->opcode)], cpu_gpr[rA(ctx->opcode)]);

#endif

}
