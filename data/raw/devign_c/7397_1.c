void tcg_gen_ld8s_i64(TCGv_i64 ret, TCGv_ptr arg2, tcg_target_long offset)

{

    tcg_gen_ld8s_i32(TCGV_LOW(ret), arg2, offset);

    tcg_gen_sari_i32(TCGV_HIGH(ret), TCGV_HIGH(ret), 31);

}
