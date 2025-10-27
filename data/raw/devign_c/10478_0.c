static void gen_store_v10_conditional(DisasContext *dc, TCGv addr, TCGv val,

                       unsigned int size, int mem_index)

{

    int l1 = gen_new_label();

    TCGv taddr = tcg_temp_local_new();

    TCGv tval = tcg_temp_local_new();

    TCGv t1 = tcg_temp_local_new();

    dc->postinc = 0;

    cris_evaluate_flags(dc);



    tcg_gen_mov_tl(taddr, addr);

    tcg_gen_mov_tl(tval, val);



    /* Store only if F flag isn't set */

    tcg_gen_andi_tl(t1, cpu_PR[PR_CCS], F_FLAG_V10);

    tcg_gen_brcondi_tl(TCG_COND_NE, t1, 0, l1);

    if (size == 1) {

        tcg_gen_qemu_st8(tval, taddr, mem_index);

    } else if (size == 2) {

        tcg_gen_qemu_st16(tval, taddr, mem_index);

    } else {

        tcg_gen_qemu_st32(tval, taddr, mem_index);

    }

    gen_set_label(l1);

    tcg_gen_shri_tl(t1, t1, 1);  /* shift F to P position */

    tcg_gen_or_tl(cpu_PR[PR_CCS], cpu_PR[PR_CCS], t1); /*P=F*/

    tcg_temp_free(t1);

    tcg_temp_free(tval);

    tcg_temp_free(taddr);

}
