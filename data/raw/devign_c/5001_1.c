static void gen_tlbsx_booke206(DisasContext *ctx)

{

#if defined(CONFIG_USER_ONLY)

    gen_inval_exception(ctx, POWERPC_EXCP_PRIV_OPC);

#else

    TCGv t0;

    if (unlikely(!ctx->mem_idx)) {

        gen_inval_exception(ctx, POWERPC_EXCP_PRIV_OPC);

        return;

    }



    if (rA(ctx->opcode)) {

        t0 = tcg_temp_new();

        tcg_gen_mov_tl(t0, cpu_gpr[rD(ctx->opcode)]);

    } else {

        t0 = tcg_const_tl(0);

    }



    tcg_gen_add_tl(t0, t0, cpu_gpr[rB(ctx->opcode)]);

    gen_helper_booke206_tlbsx(cpu_env, t0);


#endif

}