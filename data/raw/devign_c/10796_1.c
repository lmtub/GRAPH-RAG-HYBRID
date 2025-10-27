static TCGv gen_vfp_mrs(void)

{

    TCGv tmp = new_tmp();

    tcg_gen_mov_i32(tmp, cpu_F0s);

    return tmp;

}
