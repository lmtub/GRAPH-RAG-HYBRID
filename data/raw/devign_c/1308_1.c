static void gen_set_CF_bit31(TCGv var)

{

    TCGv tmp = new_tmp();

    tcg_gen_shri_i32(tmp, var, 31);

    gen_set_CF(tmp);

    dead_tmp(tmp);

}
