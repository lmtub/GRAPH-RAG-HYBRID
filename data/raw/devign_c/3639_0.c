static TCGv_i32 gen_get_asi(DisasContext *dc, int insn)

{

    int asi;



    if (IS_IMM) {

#ifdef TARGET_SPARC64

        asi = dc->asi;

#else

        gen_exception(dc, TT_ILL_INSN);

        asi = 0;

#endif

    } else {

        asi = GET_FIELD(insn, 19, 26);

    }

    return tcg_const_i32(asi);

}
