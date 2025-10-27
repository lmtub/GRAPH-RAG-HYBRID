static void gen_read_xer(TCGv dst)

{

    TCGv t0 = tcg_temp_new();

    TCGv t1 = tcg_temp_new();

    TCGv t2 = tcg_temp_new();

    tcg_gen_mov_tl(dst, cpu_xer);

    tcg_gen_shli_tl(t0, cpu_so, XER_SO);

    tcg_gen_shli_tl(t1, cpu_ov, XER_OV);

    tcg_gen_shli_tl(t2, cpu_ca, XER_CA);

    tcg_gen_or_tl(t0, t0, t1);

    tcg_gen_or_tl(dst, dst, t2);

    tcg_gen_or_tl(dst, dst, t0);

    tcg_temp_free(t0);

    tcg_temp_free(t1);

    tcg_temp_free(t2);

}
