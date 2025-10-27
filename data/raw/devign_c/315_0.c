static void gen_sraq(DisasContext *ctx)

{

    int l1 = gen_new_label();

    int l2 = gen_new_label();

    TCGv t0 = tcg_temp_new();

    TCGv t1 = tcg_temp_local_new();

    TCGv t2 = tcg_temp_local_new();

    tcg_gen_andi_tl(t2, cpu_gpr[rB(ctx->opcode)], 0x1F);

    tcg_gen_shr_tl(t0, cpu_gpr[rS(ctx->opcode)], t2);

    tcg_gen_sar_tl(t1, cpu_gpr[rS(ctx->opcode)], t2);

    tcg_gen_subfi_tl(t2, 32, t2);

    tcg_gen_shl_tl(t2, cpu_gpr[rS(ctx->opcode)], t2);

    tcg_gen_or_tl(t0, t0, t2);

    gen_store_spr(SPR_MQ, t0);

    tcg_gen_andi_tl(t0, cpu_gpr[rB(ctx->opcode)], 0x20);

    tcg_gen_brcondi_tl(TCG_COND_EQ, t2, 0, l1);

    tcg_gen_mov_tl(t2, cpu_gpr[rS(ctx->opcode)]);

    tcg_gen_sari_tl(t1, cpu_gpr[rS(ctx->opcode)], 31);

    gen_set_label(l1);

    tcg_temp_free(t0);

    tcg_gen_mov_tl(cpu_gpr[rA(ctx->opcode)], t1);

    tcg_gen_movi_tl(cpu_ca, 0);

    tcg_gen_brcondi_tl(TCG_COND_GE, t1, 0, l2);

    tcg_gen_brcondi_tl(TCG_COND_EQ, t2, 0, l2);

    tcg_gen_movi_tl(cpu_ca, 1);

    gen_set_label(l2);

    tcg_temp_free(t1);

    tcg_temp_free(t2);

    if (unlikely(Rc(ctx->opcode) != 0))

        gen_set_Rc0(ctx, cpu_gpr[rA(ctx->opcode)]);

}
