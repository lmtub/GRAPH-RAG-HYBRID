static TCGv_i64 gen_muls_i64_i32(TCGv a, TCGv b)

{

    TCGv_i64 tmp1 = tcg_temp_new_i64();

    TCGv_i64 tmp2 = tcg_temp_new_i64();



    tcg_gen_ext_i32_i64(tmp1, a);

    dead_tmp(a);

    tcg_gen_ext_i32_i64(tmp2, b);

    dead_tmp(b);

    tcg_gen_mul_i64(tmp1, tmp1, tmp2);

    tcg_temp_free_i64(tmp2);

    return tmp1;

}
