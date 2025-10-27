static TCGv neon_load_reg(int reg, int pass)

{

    TCGv tmp = new_tmp();

    tcg_gen_ld_i32(tmp, cpu_env, neon_reg_offset(reg, pass));

    return tmp;

}
