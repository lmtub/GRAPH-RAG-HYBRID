static void tcg_out_insn_3405(TCGContext *s, AArch64Insn insn, TCGType ext,

                              TCGReg rd, uint16_t half, unsigned shift)

{

    assert((shift & ~0x30) == 0);

    tcg_out32(s, insn | ext << 31 | shift << (21 - 4) | half << 5 | rd);

}
