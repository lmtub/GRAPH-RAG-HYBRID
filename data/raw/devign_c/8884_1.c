static ExitStatus gen_fbcond(DisasContext *ctx, TCGCond cond, int ra,

                             int32_t disp)

{

    TCGv cmp_tmp = tcg_temp_new();

    gen_fold_mzero(cond, cmp_tmp, load_fpr(ctx, ra));

    return gen_bcond_internal(ctx, cond, cmp_tmp, disp);

}
