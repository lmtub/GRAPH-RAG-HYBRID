static inline void gen_branch_a(DisasContext *dc, target_ulong pc1,

                                target_ulong pc2, TCGv r_cond)

{

    int l1;



    l1 = gen_new_label();



    tcg_gen_brcondi_tl(TCG_COND_EQ, r_cond, 0, l1);



    gen_goto_tb(dc, 0, pc2, pc1);



    gen_set_label(l1);

    gen_goto_tb(dc, 1, pc2 + 4, pc2 + 8);

}
