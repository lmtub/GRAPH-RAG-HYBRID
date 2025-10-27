static void gen_tlbwe_440(DisasContext *ctx)

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

    case 1:

    case 2:

        {

            TCGv_i32 t0 = tcg_const_i32(rB(ctx->opcode));

            gen_helper_440_tlbwe(cpu_env, t0, cpu_gpr[rA(ctx->opcode)],

                                 cpu_gpr[rS(ctx->opcode)]);

            tcg_temp_free_i32(t0);

        }

        break;

    default:

        gen_inval_exception(ctx, POWERPC_EXCP_INVAL_INVAL);

        break;

    }

#endif

}
