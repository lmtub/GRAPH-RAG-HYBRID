static void gen_tlbivax_booke206(DisasContext *ctx)

{

#if defined(CONFIG_USER_ONLY)

    gen_inval_exception(ctx, POWERPC_EXCP_PRIV_OPC);

#else

    TCGv t0;

    if (unlikely(ctx->pr)) {

        gen_inval_exception(ctx, POWERPC_EXCP_PRIV_OPC);

        return;

    }



    t0 = tcg_temp_new();

    gen_addr_reg_index(ctx, t0);



    gen_helper_booke206_tlbivax(cpu_env, t0);

    tcg_temp_free(t0);

#endif

}
