static void gen_op_mull_T0_T1(void)

{

    TCGv tmp1 = tcg_temp_new(TCG_TYPE_I64);

    TCGv tmp2 = tcg_temp_new(TCG_TYPE_I64);



    tcg_gen_extu_i32_i64(tmp1, cpu_T[0]);

    tcg_gen_extu_i32_i64(tmp2, cpu_T[1]);

    tcg_gen_mul_i64(tmp1, tmp1, tmp2);

    tcg_gen_trunc_i64_i32(cpu_T[0], tmp1);

    tcg_gen_shri_i64(tmp1, tmp1, 32);

    tcg_gen_trunc_i64_i32(cpu_T[1], tmp1);

}
