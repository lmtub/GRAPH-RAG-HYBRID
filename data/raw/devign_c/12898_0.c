void ioinst_handle_schm(S390CPU *cpu, uint64_t reg1, uint64_t reg2,

                        uint32_t ipb)

{

    uint8_t mbk;

    int update;

    int dct;

    CPUS390XState *env = &cpu->env;



    trace_ioinst("schm");



    if (SCHM_REG1_RES(reg1)) {

        program_interrupt(env, PGM_OPERAND, 2);

        return;

    }



    mbk = SCHM_REG1_MBK(reg1);

    update = SCHM_REG1_UPD(reg1);

    dct = SCHM_REG1_DCT(reg1);



    if (update && (reg2 & 0x000000000000001f)) {

        program_interrupt(env, PGM_OPERAND, 2);

        return;

    }



    css_do_schm(mbk, update, dct, update ? reg2 : 0);

}
