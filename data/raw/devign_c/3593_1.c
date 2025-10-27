static void gen_mulldo(DisasContext *ctx)

{

    TCGv_i64 t0 = tcg_temp_new_i64();

    TCGv_i64 t1 = tcg_temp_new_i64();



    tcg_gen_muls2_i64(t0, t1, cpu_gpr[rA(ctx->opcode)],

                      cpu_gpr[rB(ctx->opcode)]);

    tcg_gen_mov_i64(cpu_gpr[rD(ctx->opcode)], t0);



    tcg_gen_sari_i64(t0, t0, 63);

    tcg_gen_setcond_i64(TCG_COND_NE, cpu_ov, t0, t1);




    tcg_gen_or_tl(cpu_so, cpu_so, cpu_ov);



    tcg_temp_free_i64(t0);

    tcg_temp_free_i64(t1);



    if (unlikely(Rc(ctx->opcode) != 0)) {

        gen_set_Rc0(ctx, cpu_gpr[rD(ctx->opcode)]);

