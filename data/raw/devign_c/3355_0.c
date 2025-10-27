static void gen_doz(DisasContext *ctx)

{

    int l1 = gen_new_label();

    int l2 = gen_new_label();

    tcg_gen_brcond_tl(TCG_COND_GE, cpu_gpr[rB(ctx->opcode)], cpu_gpr[rA(ctx->opcode)], l1);

    tcg_gen_sub_tl(cpu_gpr[rD(ctx->opcode)], cpu_gpr[rB(ctx->opcode)], cpu_gpr[rA(ctx->opcode)]);

    tcg_gen_br(l2);

    gen_set_label(l1);

    tcg_gen_movi_tl(cpu_gpr[rD(ctx->opcode)], 0);

    gen_set_label(l2);

    if (unlikely(Rc(ctx->opcode) != 0))

        gen_set_Rc0(ctx, cpu_gpr[rD(ctx->opcode)]);

}
