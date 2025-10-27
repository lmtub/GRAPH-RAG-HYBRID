static void tcg_out_ri(TCGContext *s, int const_arg, TCGArg arg)

{

    if (const_arg) {

        assert(const_arg == 1);

        tcg_out8(s, TCG_CONST);

        tcg_out_i(s, arg);

    } else {

        tcg_out_r(s, arg);

    }

}
