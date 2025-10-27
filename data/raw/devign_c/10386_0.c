static void gen_swap_asi(DisasContext *dc, TCGv dst, TCGv src,

                         TCGv addr, int insn)

{

    TCGv_i32 r_asi, r_size, r_sign;

    TCGv_i64 s64, t64 = tcg_temp_new_i64();



    r_asi = gen_get_asi(dc, insn);

    r_size = tcg_const_i32(4);

    r_sign = tcg_const_i32(0);

    gen_helper_ld_asi(t64, cpu_env, addr, r_asi, r_size, r_sign);

    tcg_temp_free_i32(r_sign);



    s64 = tcg_temp_new_i64();

    tcg_gen_extu_tl_i64(s64, src);

    gen_helper_st_asi(cpu_env, addr, s64, r_asi, r_size);

    tcg_temp_free_i64(s64);

    tcg_temp_free_i32(r_size);

    tcg_temp_free_i32(r_asi);



    tcg_gen_trunc_i64_tl(dst, t64);

    tcg_temp_free_i64(t64);

}
