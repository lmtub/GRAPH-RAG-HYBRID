static void gen_mullwo(DisasContext *ctx)

{

    TCGv_i32 t0 = tcg_temp_new_i32();

    TCGv_i32 t1 = tcg_temp_new_i32();



    tcg_gen_trunc_tl_i32(t0, cpu_gpr[rA(ctx->opcode)]);

    tcg_gen_trunc_tl_i32(t1, cpu_gpr[rB(ctx->opcode)]);

    tcg_gen_muls2_i32(t0, t1, t0, t1);

#if defined(TARGET_PPC64)

    tcg_gen_concat_i32_i64(cpu_gpr[rD(ctx->opcode)], t0, t1);

#else

    tcg_gen_mov_i32(cpu_gpr[rD(ctx->opcode)], t0);

#endif



    tcg_gen_sari_i32(t0, t0, 31);

    tcg_gen_setcond_i32(TCG_COND_NE, t0, t0, t1);

    tcg_gen_extu_i32_tl(cpu_ov, t0);




    tcg_gen_or_tl(cpu_so, cpu_so, cpu_ov);



    tcg_temp_free_i32(t0);

    tcg_temp_free_i32(t1);

    if (unlikely(Rc(ctx->opcode) != 0))

        gen_set_Rc0(ctx, cpu_gpr[rD(ctx->opcode)]);
