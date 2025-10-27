static void patch_reloc(uint8_t *code_ptr, int type,

                        intptr_t value, intptr_t addend)

{

    value += addend;

    switch(type) {

    case R_386_PC32:

        value -= (uintptr_t)code_ptr;

        if (value != (int32_t)value) {

            tcg_abort();

        }

        *(uint32_t *)code_ptr = value;

        break;

    case R_386_PC8:

        value -= (uintptr_t)code_ptr;

        if (value != (int8_t)value) {

            tcg_abort();

        }

        *(uint8_t *)code_ptr = value;

        break;

    default:

        tcg_abort();

    }

}
