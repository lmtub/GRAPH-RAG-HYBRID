static void gen_dcbi(DisasContext *ctx)

{

#if defined(CONFIG_USER_ONLY)

    gen_inval_exception(ctx, POWERPC_EXCP_PRIV_OPC);

#else

    TCGv EA, val;

    if (unlikely(ctx->pr)) {

        gen_inval_exception(ctx, POWERPC_EXCP_PRIV_OPC);

        return;

    }

    EA = tcg_temp_new();

    gen_set_access_type(ctx, ACCESS_CACHE);

    gen_addr_reg_index(ctx, EA);

    val = tcg_temp_new();

    /* XXX: specification says this should be treated as a store by the MMU */

    gen_qemu_ld8u(ctx, val, EA);

    gen_qemu_st8(ctx, val, EA);

    tcg_temp_free(val);

    tcg_temp_free(EA);

#endif

}
