static void gen_lq(DisasContext *ctx)

{

#if defined(CONFIG_USER_ONLY)

    gen_inval_exception(ctx, POWERPC_EXCP_PRIV_OPC);

#else

    int ra, rd;

    TCGv EA;



    /* Restore CPU state */

    if (unlikely(ctx->mem_idx == 0)) {

        gen_inval_exception(ctx, POWERPC_EXCP_PRIV_OPC);

        return;

    }

    ra = rA(ctx->opcode);

    rd = rD(ctx->opcode);

    if (unlikely((rd & 1) || rd == ra)) {

        gen_inval_exception(ctx, POWERPC_EXCP_INVAL_INVAL);

        return;

    }

    if (unlikely(ctx->le_mode)) {

        /* Little-endian mode is not handled */

        gen_exception_err(ctx, POWERPC_EXCP_ALIGN, POWERPC_EXCP_ALIGN_LE);

        return;

    }

    gen_set_access_type(ctx, ACCESS_INT);

    EA = tcg_temp_new();

    gen_addr_imm_index(ctx, EA, 0x0F);

    gen_qemu_ld64(ctx, cpu_gpr[rd], EA);

    gen_addr_add(ctx, EA, EA, 8);

    gen_qemu_ld64(ctx, cpu_gpr[rd+1], EA);

    tcg_temp_free(EA);

#endif

}
