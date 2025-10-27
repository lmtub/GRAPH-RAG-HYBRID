static void patch_reloc(uint8_t *code_ptr, int type,

                        intptr_t value, intptr_t addend)

{

    uint32_t insn;

    value += addend;

    switch (type) {

    case R_SPARC_32:

        if (value != (uint32_t)value) {

            tcg_abort();

        }

        *(uint32_t *)code_ptr = value;

        break;

    case R_SPARC_WDISP16:

        value -= (intptr_t)code_ptr;

        if (!check_fit_tl(value >> 2, 16)) {

            tcg_abort();

        }

        insn = *(uint32_t *)code_ptr;

        insn &= ~INSN_OFF16(-1);

        insn |= INSN_OFF16(value);

        *(uint32_t *)code_ptr = insn;

        break;

    case R_SPARC_WDISP19:

        value -= (intptr_t)code_ptr;

        if (!check_fit_tl(value >> 2, 19)) {

            tcg_abort();

        }

        insn = *(uint32_t *)code_ptr;

        insn &= ~INSN_OFF19(-1);

        insn |= INSN_OFF19(value);

        *(uint32_t *)code_ptr = insn;

        break;

    default:

        tcg_abort();

    }

}
