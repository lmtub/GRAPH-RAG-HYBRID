static void gen_storeq_reg(DisasContext *s, int rlow, int rhigh, TCGv_i64 val)

{

    TCGv tmp;

    tmp = new_tmp();

    tcg_gen_trunc_i64_i32(tmp, val);

    store_reg(s, rlow, tmp);

    tmp = new_tmp();

    tcg_gen_shri_i64(val, val, 32);

    tcg_gen_trunc_i64_i32(tmp, val);

    store_reg(s, rhigh, tmp);

}
