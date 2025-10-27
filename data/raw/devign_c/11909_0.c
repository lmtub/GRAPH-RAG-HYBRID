static inline void tcg_out_goto(TCGContext *s, tcg_insn_unit *target)

{

    ptrdiff_t offset = target - s->code_ptr;

    assert(offset == sextract64(offset, 0, 26));

    tcg_out_insn(s, 3206, B, offset);

}
