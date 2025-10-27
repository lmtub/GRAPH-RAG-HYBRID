static void tgen_ext16u(TCGContext *s, TCGType type, TCGReg dest, TCGReg src)

{

    if (facilities & FACILITY_EXT_IMM) {

        tcg_out_insn(s, RRE, LLGHR, dest, src);

        return;

    }



    if (dest == src) {

        tcg_out_movi(s, type, TCG_TMP0, 0xffff);

        src = TCG_TMP0;

    } else {

        tcg_out_movi(s, type, dest, 0xffff);

    }

    if (type == TCG_TYPE_I32) {

        tcg_out_insn(s, RR, NR, dest, src);

    } else {

        tcg_out_insn(s, RRE, NGR, dest, src);

    }

}
