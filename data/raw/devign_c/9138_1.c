static void gen_tlbre_40x(DisasContext *ctx)

{

#if defined(CONFIG_USER_ONLY)

    gen_inval_exception(ctx, POWERPC_EXCP_PRIV_OPC);

#else

    if (unlikely(ctx->pr)) {

        gen_inval_exception(ctx, POWERPC_EXCP_PRIV_OPC);

        return;

    }

    switch (rB(ctx->opcode)) {

    case 0:

        gen_helper_4xx_tlbre_hi(cpu_gpr[rD(ctx->opcode)], cpu_env,

                                cpu_gpr[rA(ctx->opcode)]);

        break;

    case 1:

        gen_helper_4xx_tlbre_lo(cpu_gpr[rD(ctx->opcode)], cpu_env,

                                cpu_gpr[rA(ctx->opcode)]);

        break;

    default:

        gen_inval_exception(ctx, POWERPC_EXCP_INVAL_INVAL);

        break;

    }

#endif

}
