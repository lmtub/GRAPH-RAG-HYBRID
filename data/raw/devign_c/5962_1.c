static void gen_addq_lo(DisasContext *s, TCGv_i64 val, int rlow)

{

    TCGv_i64 tmp;

    TCGv tmp2;



    /* Load value and extend to 64 bits.  */

    tmp = tcg_temp_new_i64();

    tmp2 = load_reg(s, rlow);

    tcg_gen_extu_i32_i64(tmp, tmp2);

    dead_tmp(tmp2);

    tcg_gen_add_i64(val, val, tmp);

    tcg_temp_free_i64(tmp);

}
