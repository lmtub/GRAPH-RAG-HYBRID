static inline void reloc_pc26(tcg_insn_unit *code_ptr, tcg_insn_unit *target)

{

    ptrdiff_t offset = target - code_ptr;

    assert(offset == sextract64(offset, 0, 26));

    /* read instruction, mask away previous PC_REL26 parameter contents,

       set the proper offset, then write back the instruction. */

    *code_ptr = deposit32(*code_ptr, 0, 26, offset);

}
