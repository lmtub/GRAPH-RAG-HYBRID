static void gen_rfid(DisasContext *ctx)

{

#if defined(CONFIG_USER_ONLY)

    gen_inval_exception(ctx, POWERPC_EXCP_PRIV_OPC);

#else

    /* Restore CPU state */

    if (unlikely(!ctx->mem_idx)) {

        gen_inval_exception(ctx, POWERPC_EXCP_PRIV_OPC);

        return;

    }


    gen_helper_rfid();

    gen_sync_exception(ctx);

#endif

}