static void gen_tlbwe_booke206(DisasContext *ctx)

{

#if defined(CONFIG_USER_ONLY)

    gen_inval_exception(ctx, POWERPC_EXCP_PRIV_OPC);

#else

    if (unlikely(ctx->pr)) {

        gen_inval_exception(ctx, POWERPC_EXCP_PRIV_OPC);

        return;

    }

    gen_update_nip(ctx, ctx->nip - 4);

    gen_helper_booke206_tlbwe(cpu_env);

#endif

}
