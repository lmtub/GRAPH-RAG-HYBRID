void tcg_gen_brcondi_i64(TCGCond cond, TCGv_i64 arg1, int64_t arg2, int label)

{

    if (cond == TCG_COND_ALWAYS) {

        tcg_gen_br(label);

    } else if (cond != TCG_COND_NEVER) {

        TCGv_i64 t0 = tcg_const_i64(arg2);

        tcg_gen_brcond_i64(cond, arg1, t0, label);

        tcg_temp_free_i64(t0);

    }

}
