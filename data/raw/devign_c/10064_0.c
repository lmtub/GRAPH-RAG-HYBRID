static ExitStatus gen_bcond_internal(DisasContext *ctx, TCGCond cond,

                                     TCGv cmp, int32_t disp)

{

    uint64_t dest = ctx->pc + (disp << 2);

    int lab_true = gen_new_label();



    if (use_goto_tb(ctx, dest)) {

        tcg_gen_brcondi_i64(cond, cmp, 0, lab_true);



        tcg_gen_goto_tb(0);

        tcg_gen_movi_i64(cpu_pc, ctx->pc);

        tcg_gen_exit_tb((uintptr_t)ctx->tb);



        gen_set_label(lab_true);

        tcg_gen_goto_tb(1);

        tcg_gen_movi_i64(cpu_pc, dest);

        tcg_gen_exit_tb((uintptr_t)ctx->tb + 1);



        return EXIT_GOTO_TB;

    } else {

        TCGv_i64 z = tcg_const_i64(0);

        TCGv_i64 d = tcg_const_i64(dest);

        TCGv_i64 p = tcg_const_i64(ctx->pc);



        tcg_gen_movcond_i64(cond, cpu_pc, cmp, z, d, p);



        tcg_temp_free_i64(z);

        tcg_temp_free_i64(d);

        tcg_temp_free_i64(p);

        return EXIT_PC_UPDATED;

    }

}
