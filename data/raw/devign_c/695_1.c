static TCGv_i64 gen_subq_msw(TCGv_i64 a, TCGv b)

{

    TCGv_i64 tmp64 = tcg_temp_new_i64();



    tcg_gen_extu_i32_i64(tmp64, b);

    dead_tmp(b);

    tcg_gen_shli_i64(tmp64, tmp64, 32);

    tcg_gen_sub_i64(a, tmp64, a);



    tcg_temp_free_i64(tmp64);

    return a;

}
