static void gen_rfci(DisasContext *ctx)

{

#if defined(CONFIG_USER_ONLY)

    gen_inval_exception(ctx, POWERPC_EXCP_PRIV_OPC);

#else

    if (unlikely(ctx->pr)) {

        gen_inval_exception(ctx, POWERPC_EXCP_PRIV_OPC);

        return;

    }

    /* Restore CPU state */

    gen_helper_rfci(cpu_env);

    gen_sync_exception(ctx);

#endif

}
