static inline void tcg_out_adr(TCGContext *s, TCGReg rd, void *target)

{

    ptrdiff_t offset = tcg_pcrel_diff(s, target);

    assert(offset == sextract64(offset, 0, 21));

    tcg_out_insn(s, 3406, ADR, rd, offset);

}
