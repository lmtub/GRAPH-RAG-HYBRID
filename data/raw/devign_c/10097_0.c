static inline void gen_evsel(DisasContext *ctx)

{

    int l1 = gen_new_label();

    int l2 = gen_new_label();

    int l3 = gen_new_label();

    int l4 = gen_new_label();

    TCGv_i32 t0 = tcg_temp_local_new_i32();

    tcg_gen_andi_i32(t0, cpu_crf[ctx->opcode & 0x07], 1 << 3);

    tcg_gen_brcondi_i32(TCG_COND_EQ, t0, 0, l1);

    tcg_gen_mov_tl(cpu_gprh[rD(ctx->opcode)], cpu_gprh[rA(ctx->opcode)]);

    tcg_gen_br(l2);

    gen_set_label(l1);

    tcg_gen_mov_tl(cpu_gprh[rD(ctx->opcode)], cpu_gprh[rB(ctx->opcode)]);

    gen_set_label(l2);

    tcg_gen_andi_i32(t0, cpu_crf[ctx->opcode & 0x07], 1 << 2);

    tcg_gen_brcondi_i32(TCG_COND_EQ, t0, 0, l3);

    tcg_gen_mov_tl(cpu_gpr[rD(ctx->opcode)], cpu_gpr[rA(ctx->opcode)]);

    tcg_gen_br(l4);

    gen_set_label(l3);

    tcg_gen_mov_tl(cpu_gpr[rD(ctx->opcode)], cpu_gpr[rB(ctx->opcode)]);

    gen_set_label(l4);

    tcg_temp_free_i32(t0);

}
