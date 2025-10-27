static void disas_pc_rel_adr(DisasContext *s, uint32_t insn)

{

    unsigned int page, rd;

    uint64_t base;

    int64_t offset;



    page = extract32(insn, 31, 1);

    /* SignExtend(immhi:immlo) -> offset */

    offset = ((int64_t)sextract32(insn, 5, 19) << 2) | extract32(insn, 29, 2);

    rd = extract32(insn, 0, 5);

    base = s->pc - 4;



    if (page) {

        /* ADRP (page based) */

        base &= ~0xfff;

        offset <<= 12;

    }



    tcg_gen_movi_i64(cpu_reg(s, rd), base + offset);

}
