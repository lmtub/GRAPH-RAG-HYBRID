static void gen_nabs(DisasContext *ctx)

{

    int l1 = gen_new_label();

    int l2 = gen_new_label();

    tcg_gen_brcondi_tl(TCG_COND_GT, cpu_gpr[rA(ctx->opcode)], 0, l1);

    tcg_gen_mov_tl(cpu_gpr[rD(ctx->opcode)], cpu_gpr[rA(ctx->opcode)]);

    tcg_gen_br(l2);

    gen_set_label(l1);

    tcg_gen_neg_tl(cpu_gpr[rD(ctx->opcode)], cpu_gpr[rA(ctx->opcode)]);

    gen_set_label(l2);

    if (unlikely(Rc(ctx->opcode) != 0))

        gen_set_Rc0(ctx, cpu_gpr[rD(ctx->opcode)]);

}
