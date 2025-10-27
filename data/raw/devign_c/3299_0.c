static void t_gen_cris_dstep(TCGv d, TCGv a, TCGv b)

{

    int l1;



    l1 = gen_new_label();



    /*

     * d <<= 1

     * if (d >= s)

     *    d -= s;

     */

    tcg_gen_shli_tl(d, a, 1);

    tcg_gen_brcond_tl(TCG_COND_LTU, d, b, l1);

    tcg_gen_sub_tl(d, d, b);

    gen_set_label(l1);

}
