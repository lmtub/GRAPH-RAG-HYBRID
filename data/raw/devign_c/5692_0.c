static void t_gen_muls(TCGv d, TCGv d2, TCGv a, TCGv b)

{

	TCGv t0, t1;



	t0 = tcg_temp_new(TCG_TYPE_I64);

	t1 = tcg_temp_new(TCG_TYPE_I64);



	tcg_gen_ext32s_i64(t0, a);

	tcg_gen_ext32s_i64(t1, b);

	tcg_gen_mul_i64(t0, t0, t1);



	tcg_gen_trunc_i64_i32(d, t0);

	tcg_gen_shri_i64(t0, t0, 32);

	tcg_gen_trunc_i64_i32(d2, t0);



	tcg_temp_free(t0);

	tcg_temp_free(t1);

}
