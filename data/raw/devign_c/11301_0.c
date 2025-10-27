static void gen_neon_zip_u16(TCGv t0, TCGv t1)

{

    TCGv tmp, tmp2;



    tmp = new_tmp();

    tmp2 = new_tmp();



    tcg_gen_andi_i32(tmp, t0, 0xffff);

    tcg_gen_shli_i32(tmp2, t1, 16);

    tcg_gen_or_i32(tmp, tmp, tmp2);

    tcg_gen_andi_i32(t1, t1, 0xffff0000);

    tcg_gen_shri_i32(tmp2, t0, 16);

    tcg_gen_or_i32(t1, t1, tmp2);

    tcg_gen_mov_i32(t0, tmp);



    dead_tmp(tmp2);

    dead_tmp(tmp);

}
