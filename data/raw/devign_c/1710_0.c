static void gen_isel(DisasContext *ctx)

{

    int l1, l2;

    uint32_t bi = rC(ctx->opcode);

    uint32_t mask;

    TCGv_i32 t0;



    l1 = gen_new_label();

    l2 = gen_new_label();



    mask = 0x08 >> (bi & 0x03);

    t0 = tcg_temp_new_i32();

    tcg_gen_andi_i32(t0, cpu_crf[bi >> 2], mask);

    tcg_gen_brcondi_i32(TCG_COND_EQ, t0, 0, l1);

    if (rA(ctx->opcode) == 0)

        tcg_gen_movi_tl(cpu_gpr[rD(ctx->opcode)], 0);

    else

        tcg_gen_mov_tl(cpu_gpr[rD(ctx->opcode)], cpu_gpr[rA(ctx->opcode)]);

    tcg_gen_br(l2);

    gen_set_label(l1);

    tcg_gen_mov_tl(cpu_gpr[rD(ctx->opcode)], cpu_gpr[rB(ctx->opcode)]);

    gen_set_label(l2);

    tcg_temp_free_i32(t0);

}
