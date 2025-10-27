static void gen_slq(DisasContext *ctx)

{

    int l1 = gen_new_label();

    TCGv t0 = tcg_temp_new();

    TCGv t1 = tcg_temp_new();

    tcg_gen_andi_tl(t1, cpu_gpr[rB(ctx->opcode)], 0x1F);

    tcg_gen_shl_tl(t0, cpu_gpr[rS(ctx->opcode)], t1);

    tcg_gen_subfi_tl(t1, 32, t1);

    tcg_gen_shr_tl(t1, cpu_gpr[rS(ctx->opcode)], t1);

    tcg_gen_or_tl(t1, t0, t1);

    gen_store_spr(SPR_MQ, t1);

    tcg_gen_andi_tl(t1, cpu_gpr[rB(ctx->opcode)], 0x20);

    tcg_gen_mov_tl(cpu_gpr[rA(ctx->opcode)], t0);

    tcg_gen_brcondi_tl(TCG_COND_EQ, t1, 0, l1);

    tcg_gen_movi_tl(cpu_gpr[rA(ctx->opcode)], 0);

    gen_set_label(l1);

    tcg_temp_free(t0);

    tcg_temp_free(t1);

    if (unlikely(Rc(ctx->opcode) != 0))

        gen_set_Rc0(ctx, cpu_gpr[rA(ctx->opcode)]);

}
