static void tcg_out_brcond_i32(TCGContext *s, TCGCond cond, TCGReg arg1,

                               int32_t arg2, int const_arg2, int label)

{

    tcg_out_cmp(s, arg1, arg2, const_arg2);

    tcg_out_bpcc(s, tcg_cond_to_bcond[cond], BPCC_ICC | BPCC_PT, label);

    tcg_out_nop(s);

}
