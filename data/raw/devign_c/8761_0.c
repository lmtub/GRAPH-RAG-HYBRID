void ioinst_handle_sal(S390CPU *cpu, uint64_t reg1)

{

    /* We do not provide address limit checking, so let's suppress it. */

    if (SAL_REG1_INVALID(reg1) || reg1 & 0x000000000000ffff) {

        program_interrupt(&cpu->env, PGM_OPERAND, 2);

    }

}
