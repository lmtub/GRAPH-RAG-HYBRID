static void neon_store_reg(int reg, int pass, TCGv var)

{

    tcg_gen_st_i32(var, cpu_env, neon_reg_offset(reg, pass));

    dead_tmp(var);

}
