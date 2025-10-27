static inline void gen_movcf_ps(DisasContext *ctx, int fs, int fd,

                                int cc, int tf)

{

    int cond;

    TCGv_i32 t0 = tcg_temp_new_i32();

    int l1 = gen_new_label();

    int l2 = gen_new_label();



    if (tf)

        cond = TCG_COND_EQ;

    else

        cond = TCG_COND_NE;



    tcg_gen_andi_i32(t0, fpu_fcr31, 1 << get_fp_bit(cc));

    tcg_gen_brcondi_i32(cond, t0, 0, l1);

    gen_load_fpr32(t0, fs);

    gen_store_fpr32(t0, fd);

    gen_set_label(l1);



    tcg_gen_andi_i32(t0, fpu_fcr31, 1 << get_fp_bit(cc+1));

    tcg_gen_brcondi_i32(cond, t0, 0, l2);

    gen_load_fpr32h(ctx, t0, fs);

    gen_store_fpr32h(ctx, t0, fd);

    tcg_temp_free_i32(t0);

    gen_set_label(l2);

}
