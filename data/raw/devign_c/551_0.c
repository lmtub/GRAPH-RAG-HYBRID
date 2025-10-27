static void gen_stda_asi(DisasContext *dc, TCGv hi, TCGv addr,

                         int insn, int rd)

{

    TCGv_i32 r_asi, r_size;

    TCGv lo = gen_load_gpr(dc, rd + 1);

    TCGv_i64 t64 = tcg_temp_new_i64();



    tcg_gen_concat_tl_i64(t64, lo, hi);

    r_asi = gen_get_asi(dc, insn);

    r_size = tcg_const_i32(8);

    gen_helper_st_asi(cpu_env, addr, t64, r_asi, r_size);

    tcg_temp_free_i32(r_size);

    tcg_temp_free_i32(r_asi);

    tcg_temp_free_i64(t64);

}
