static void gen_sraiq(DisasContext *ctx)

{

    int sh = SH(ctx->opcode);

    int l1 = gen_new_label();

    TCGv t0 = tcg_temp_new();

    TCGv t1 = tcg_temp_new();

    tcg_gen_shri_tl(t0, cpu_gpr[rS(ctx->opcode)], sh);

    tcg_gen_shli_tl(t1, cpu_gpr[rS(ctx->opcode)], 32 - sh);

    tcg_gen_or_tl(t0, t0, t1);

    gen_store_spr(SPR_MQ, t0);

    tcg_gen_movi_tl(cpu_ca, 0);

    tcg_gen_brcondi_tl(TCG_COND_EQ, t1, 0, l1);

    tcg_gen_brcondi_tl(TCG_COND_GE, cpu_gpr[rS(ctx->opcode)], 0, l1);

    tcg_gen_movi_tl(cpu_ca, 1);

    gen_set_label(l1);

    tcg_gen_sari_tl(cpu_gpr[rA(ctx->opcode)], cpu_gpr[rS(ctx->opcode)], sh);

    tcg_temp_free(t0);

    tcg_temp_free(t1);

    if (unlikely(Rc(ctx->opcode) != 0))

        gen_set_Rc0(ctx, cpu_gpr[rA(ctx->opcode)]);

}
