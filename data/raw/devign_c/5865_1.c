static void gen_tlbsx_40x(DisasContext *ctx)

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

    gen_helper_4xx_tlbsx(cpu_gpr[rD(ctx->opcode)], cpu_env, t0);

    tcg_temp_free(t0);

    if (Rc(ctx->opcode)) {

        TCGLabel *l1 = gen_new_label();

        tcg_gen_trunc_tl_i32(cpu_crf[0], cpu_so);

        tcg_gen_brcondi_tl(TCG_COND_EQ, cpu_gpr[rD(ctx->opcode)], -1, l1);

        tcg_gen_ori_i32(cpu_crf[0], cpu_crf[0], 0x02);

        gen_set_label(l1);

    }

#endif

}
