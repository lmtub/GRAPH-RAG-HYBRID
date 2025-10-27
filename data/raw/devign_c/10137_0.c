static void gen_abso(DisasContext *ctx)

{

    int l1 = gen_new_label();

    int l2 = gen_new_label();

    int l3 = gen_new_label();

    /* Start with XER OV disabled, the most likely case */

    tcg_gen_movi_tl(cpu_ov, 0);

    tcg_gen_brcondi_tl(TCG_COND_GE, cpu_gpr[rA(ctx->opcode)], 0, l2);

    tcg_gen_brcondi_tl(TCG_COND_NE, cpu_gpr[rA(ctx->opcode)], 0x80000000, l1);

    tcg_gen_movi_tl(cpu_ov, 1);

    tcg_gen_movi_tl(cpu_so, 1);

    tcg_gen_br(l2);

    gen_set_label(l1);

    tcg_gen_neg_tl(cpu_gpr[rD(ctx->opcode)], cpu_gpr[rA(ctx->opcode)]);

    tcg_gen_br(l3);

    gen_set_label(l2);

    tcg_gen_mov_tl(cpu_gpr[rD(ctx->opcode)], cpu_gpr[rA(ctx->opcode)]);

    gen_set_label(l3);

    if (unlikely(Rc(ctx->opcode) != 0))

        gen_set_Rc0(ctx, cpu_gpr[rD(ctx->opcode)]);

}
