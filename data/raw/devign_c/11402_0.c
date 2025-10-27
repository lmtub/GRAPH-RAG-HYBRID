static void gen_load_fpr32h(TCGv_i32 t, int reg)

{

    TCGv_i64 t64 = tcg_temp_new_i64();

    tcg_gen_shri_i64(t64, fpu_f64[reg], 32);

    tcg_gen_trunc_i64_i32(t, t64);

    tcg_temp_free_i64(t64);

}
