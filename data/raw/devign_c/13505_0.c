static void eval_cond_jmp(DisasContext *dc, TCGv pc_true, TCGv pc_false)

{

    int l1;



    l1 = gen_new_label();

    /* Conditional jmp.  */

    tcg_gen_mov_tl(cpu_SR[SR_PC], pc_false);

    tcg_gen_brcondi_tl(TCG_COND_EQ, env_btaken, 0, l1);

    tcg_gen_mov_tl(cpu_SR[SR_PC], pc_true);

    gen_set_label(l1);

}
