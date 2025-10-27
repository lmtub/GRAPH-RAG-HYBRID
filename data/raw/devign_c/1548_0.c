static void patch_reloc(tcg_insn_unit *code_ptr, int type,

                        intptr_t value, intptr_t addend)

{

    assert(type == R_ARM_PC24);

    assert(addend == 0);

    reloc_pc24(code_ptr, (tcg_insn_unit *)value);

}
