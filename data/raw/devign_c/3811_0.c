static void tcg_out_insn_3314(TCGContext *s, AArch64Insn insn,

                              TCGReg r1, TCGReg r2, TCGReg rn,

                              tcg_target_long ofs, bool pre, bool w)

{

    insn |= 1u << 31; /* ext */

    insn |= pre << 24;

    insn |= w << 23;



    assert(ofs >= -0x200 && ofs < 0x200 && (ofs & 7) == 0);

    insn |= (ofs & (0x7f << 3)) << (15 - 3);



    tcg_out32(s, insn | r2 << 10 | rn << 5 | r1);

}
