static void tcg_out_brcond(TCGContext *s, TCGMemOp ext, TCGCond c, TCGArg a,

                           TCGArg b, bool b_const, TCGLabel *l)

{

    intptr_t offset;

    bool need_cmp;



    if (b_const && b == 0 && (c == TCG_COND_EQ || c == TCG_COND_NE)) {

        need_cmp = false;

    } else {

        need_cmp = true;

        tcg_out_cmp(s, ext, a, b, b_const);

    }



    if (!l->has_value) {

        tcg_out_reloc(s, s->code_ptr, R_AARCH64_CONDBR19, l, 0);

        offset = tcg_in32(s) >> 5;

    } else {

        offset = l->u.value_ptr - s->code_ptr;

        assert(offset == sextract64(offset, 0, 19));

    }



    if (need_cmp) {

        tcg_out_insn(s, 3202, B_C, c, offset);

    } else if (c == TCG_COND_EQ) {

        tcg_out_insn(s, 3201, CBZ, ext, a, offset);

    } else {

        tcg_out_insn(s, 3201, CBNZ, ext, a, offset);

    }

}
