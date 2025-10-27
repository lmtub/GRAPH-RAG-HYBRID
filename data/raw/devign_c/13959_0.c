static void gen_mulo(DisasContext *ctx)

{

    int l1 = gen_new_label();

    TCGv_i64 t0 = tcg_temp_new_i64();

    TCGv_i64 t1 = tcg_temp_new_i64();

    TCGv t2 = tcg_temp_new();

    /* Start with XER OV disabled, the most likely case */

    tcg_gen_movi_tl(cpu_ov, 0);

    tcg_gen_extu_tl_i64(t0, cpu_gpr[rA(ctx->opcode)]);

    tcg_gen_extu_tl_i64(t1, cpu_gpr[rB(ctx->opcode)]);

    tcg_gen_mul_i64(t0, t0, t1);

    tcg_gen_trunc_i64_tl(t2, t0);

    gen_store_spr(SPR_MQ, t2);

    tcg_gen_shri_i64(t1, t0, 32);

    tcg_gen_trunc_i64_tl(cpu_gpr[rD(ctx->opcode)], t1);

    tcg_gen_ext32s_i64(t1, t0);

    tcg_gen_brcond_i64(TCG_COND_EQ, t0, t1, l1);

    tcg_gen_movi_tl(cpu_ov, 1);

    tcg_gen_movi_tl(cpu_so, 1);

    gen_set_label(l1);

    tcg_temp_free_i64(t0);

    tcg_temp_free_i64(t1);

    tcg_temp_free(t2);

    if (unlikely(Rc(ctx->opcode) != 0))

        gen_set_Rc0(ctx, cpu_gpr[rD(ctx->opcode)]);

}
