static void tcg_out_brcond32(TCGContext *s, TCGCond cond,

                             TCGArg arg1, TCGArg arg2, int const_arg2,

                             int label_index, int small)

{

    tcg_out_cmp(s, arg1, arg2, const_arg2, 0);

    tcg_out_jxx(s, tcg_cond_to_jcc[cond], label_index, small);

}
