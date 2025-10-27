void ioinst_handle_rchp(S390CPU *cpu, uint64_t reg1)

{

    int cc;

    uint8_t cssid;

    uint8_t chpid;

    int ret;

    CPUS390XState *env = &cpu->env;



    if (RCHP_REG1_RES(reg1)) {

        program_interrupt(env, PGM_OPERAND, 2);

        return;

    }



    cssid = RCHP_REG1_CSSID(reg1);

    chpid = RCHP_REG1_CHPID(reg1);



    trace_ioinst_chp_id("rchp", cssid, chpid);



    ret = css_do_rchp(cssid, chpid);



    switch (ret) {

    case -ENODEV:

        cc = 3;

        break;

    case -EBUSY:

        cc = 2;

        break;

    case 0:

        cc = 0;

        break;

    default:

        /* Invalid channel subsystem. */

        program_interrupt(env, PGM_OPERAND, 2);

        return;

    }

    setcc(cpu, cc);

}
