static inline void gen_movcf_ps (int fs, int fd, int cc, int tf)

{

    int cond;

    TCGv r_tmp1 = tcg_temp_local_new(TCG_TYPE_I32);

    TCGv r_tmp2 = tcg_temp_local_new(TCG_TYPE_I32);

    TCGv fp0 = tcg_temp_local_new(TCG_TYPE_I32);

    TCGv fph0 = tcg_temp_local_new(TCG_TYPE_I32);

    TCGv fp1 = tcg_temp_local_new(TCG_TYPE_I32);

    TCGv fph1 = tcg_temp_local_new(TCG_TYPE_I32);

    int l1 = gen_new_label();

    int l2 = gen_new_label();



    if (tf)

        cond = TCG_COND_EQ;

    else

        cond = TCG_COND_NE;



    gen_load_fpr32(fp0, fs);

    gen_load_fpr32h(fph0, fs);

    gen_load_fpr32(fp1, fd);

    gen_load_fpr32h(fph1, fd);

    get_fp_cond(r_tmp1);

    tcg_gen_shri_i32(r_tmp1, r_tmp1, cc);

    tcg_gen_andi_i32(r_tmp2, r_tmp1, 0x1);

    tcg_gen_brcondi_i32(cond, r_tmp2, 0, l1);

    tcg_gen_mov_i32(fp1, fp0);

    tcg_temp_free(fp0);

    gen_set_label(l1);

    tcg_gen_andi_i32(r_tmp2, r_tmp1, 0x2);

    tcg_gen_brcondi_i32(cond, r_tmp2, 0, l2);

    tcg_gen_mov_i32(fph1, fph0);

    tcg_temp_free(fph0);

    gen_set_label(l2);

    tcg_temp_free(r_tmp1);

    tcg_temp_free(r_tmp2);

    gen_store_fpr32(fp1, fd);

    gen_store_fpr32h(fph1, fd);

    tcg_temp_free(fp1);

    tcg_temp_free(fph1);

}
