static TCGv_i32 read_fp_sreg(DisasContext *s, int reg)

{

    TCGv_i32 v = tcg_temp_new_i32();



    tcg_gen_ld_i32(v, cpu_env, fp_reg_offset(reg, MO_32));

    return v;

}
