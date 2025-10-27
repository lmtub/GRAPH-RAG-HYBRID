static void tcg_out_r(TCGContext *s, TCGArg t0)

{

    assert(t0 < TCG_TARGET_NB_REGS);

    tcg_out8(s, t0);

}
