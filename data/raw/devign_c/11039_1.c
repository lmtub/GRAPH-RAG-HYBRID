static void gen_smul_dual(TCGv a, TCGv b)

{

    TCGv tmp1 = new_tmp();

    TCGv tmp2 = new_tmp();

    tcg_gen_ext16s_i32(tmp1, a);

    tcg_gen_ext16s_i32(tmp2, b);

    tcg_gen_mul_i32(tmp1, tmp1, tmp2);

    dead_tmp(tmp2);

    tcg_gen_sari_i32(a, a, 16);

    tcg_gen_sari_i32(b, b, 16);

    tcg_gen_mul_i32(b, b, a);

    tcg_gen_mov_i32(a, tmp1);

    dead_tmp(tmp1);

}
