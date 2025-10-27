static void gen_neon_trn_u16(TCGv t0, TCGv t1)

{

    TCGv rd, tmp;



    rd = new_tmp();

    tmp = new_tmp();



    tcg_gen_shli_i32(rd, t0, 16);

    tcg_gen_andi_i32(tmp, t1, 0xffff);

    tcg_gen_or_i32(rd, rd, tmp);

    tcg_gen_shri_i32(t1, t1, 16);

    tcg_gen_andi_i32(tmp, t0, 0xffff0000);

    tcg_gen_or_i32(t1, t1, tmp);

    tcg_gen_mov_i32(t0, rd);



    dead_tmp(tmp);

    dead_tmp(rd);

}
