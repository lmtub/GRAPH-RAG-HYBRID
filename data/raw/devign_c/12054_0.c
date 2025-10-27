static void gen_imull(TCGv a, TCGv b)

{

    TCGv tmp1 = tcg_temp_new(TCG_TYPE_I64);

    TCGv tmp2 = tcg_temp_new(TCG_TYPE_I64);



    tcg_gen_ext_i32_i64(tmp1, a);

    tcg_gen_ext_i32_i64(tmp2, b);

    tcg_gen_mul_i64(tmp1, tmp1, tmp2);

    tcg_gen_trunc_i64_i32(a, tmp1);

    tcg_gen_shri_i64(tmp1, tmp1, 32);

    tcg_gen_trunc_i64_i32(b, tmp1);

}
