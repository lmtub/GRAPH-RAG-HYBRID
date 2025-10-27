void tcg_gen_brcond_i32(TCGCond cond, TCGv_i32 arg1, TCGv_i32 arg2, int label)

{

    if (cond == TCG_COND_ALWAYS) {

        tcg_gen_br(label);

    } else if (cond != TCG_COND_NEVER) {

        tcg_gen_op4ii_i32(INDEX_op_brcond_i32, arg1, arg2, cond, label);

    }

}
