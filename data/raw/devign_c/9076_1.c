static void gen_tlbwe_40x(DisasContext *ctx)

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

        gen_helper_4xx_tlbwe_hi(cpu_env, cpu_gpr[rA(ctx->opcode)],

                                cpu_gpr[rS(ctx->opcode)]);

        break;

    case 1:

        gen_helper_4xx_tlbwe_lo(cpu_env, cpu_gpr[rA(ctx->opcode)],

                                cpu_gpr[rS(ctx->opcode)]);

        break;

    default:

        gen_inval_exception(ctx, POWERPC_EXCP_INVAL_INVAL);

        break;

    }

#endif

}
