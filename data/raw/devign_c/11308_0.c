static void gen_store_fpr32h(TCGv_i32 t, int reg)

{

    TCGv_i64 t64 = tcg_temp_new_i64();

    tcg_gen_extu_i32_i64(t64, t);

    tcg_gen_deposit_i64(fpu_f64[reg], fpu_f64[reg], t64, 32, 32);

    tcg_temp_free_i64(t64);

}
