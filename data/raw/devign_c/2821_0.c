static void gen_sllq(DisasContext *ctx)

{

    int l1 = gen_new_label();

    int l2 = gen_new_label();

    TCGv t0 = tcg_temp_local_new();

    TCGv t1 = tcg_temp_local_new();

    TCGv t2 = tcg_temp_local_new();

    tcg_gen_andi_tl(t2, cpu_gpr[rB(ctx->opcode)], 0x1F);

    tcg_gen_movi_tl(t1, 0xFFFFFFFF);

    tcg_gen_shl_tl(t1, t1, t2);

    tcg_gen_andi_tl(t0, cpu_gpr[rB(ctx->opcode)], 0x20);

    tcg_gen_brcondi_tl(TCG_COND_EQ, t0, 0, l1);

    gen_load_spr(t0, SPR_MQ);

    tcg_gen_and_tl(cpu_gpr[rA(ctx->opcode)], t0, t1);

    tcg_gen_br(l2);

    gen_set_label(l1);

    tcg_gen_shl_tl(t0, cpu_gpr[rS(ctx->opcode)], t2);

    gen_load_spr(t2, SPR_MQ);

    tcg_gen_andc_tl(t1, t2, t1);

    tcg_gen_or_tl(cpu_gpr[rA(ctx->opcode)], t0, t1);

    gen_set_label(l2);

    tcg_temp_free(t0);

    tcg_temp_free(t1);

    tcg_temp_free(t2);

    if (unlikely(Rc(ctx->opcode) != 0))

        gen_set_Rc0(ctx, cpu_gpr[rA(ctx->opcode)]);

}
