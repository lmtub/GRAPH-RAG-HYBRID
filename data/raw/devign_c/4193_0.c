int ioinst_handle_schm(CPUS390XState *env, uint64_t reg1, uint64_t reg2,

                       uint32_t ipb)

{

    uint8_t mbk;

    int update;

    int dct;



    trace_ioinst("schm");



    if (SCHM_REG1_RES(reg1)) {

        program_interrupt(env, PGM_OPERAND, 2);

        return -EIO;

    }



    mbk = SCHM_REG1_MBK(reg1);

    update = SCHM_REG1_UPD(reg1);

    dct = SCHM_REG1_DCT(reg1);



    if (update && (reg2 & 0x0000000000000fff)) {

        program_interrupt(env, PGM_OPERAND, 2);

        return -EIO;

    }



    css_do_schm(mbk, update, dct, update ? reg2 : 0);



    return 0;

}
