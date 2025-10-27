static inline void tcg_out_ld_ptr(TCGContext *s, TCGReg ret, uintptr_t arg)

{

    TCGReg base = TCG_REG_G0;

    if (!check_fit_tl(arg, 10)) {

        tcg_out_movi(s, TCG_TYPE_PTR, ret, arg & ~0x3ff);

        base = ret;

    }

    tcg_out_ld(s, TCG_TYPE_PTR, ret, base, arg & 0x3ff);

}
