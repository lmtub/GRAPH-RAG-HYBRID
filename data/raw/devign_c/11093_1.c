void ioinst_handle_rsch(S390CPU *cpu, uint64_t reg1)

{

    int cssid, ssid, schid, m;

    SubchDev *sch;

    int ret = -ENODEV;

    int cc;



    if (ioinst_disassemble_sch_ident(reg1, &m, &cssid, &ssid, &schid)) {

        program_interrupt(&cpu->env, PGM_OPERAND, 4);

        return;

    }

    trace_ioinst_sch_id("rsch", cssid, ssid, schid);

    sch = css_find_subch(m, cssid, ssid, schid);

    if (sch && css_subch_visible(sch)) {

        ret = css_do_rsch(sch);

    }

    switch (ret) {

    case -ENODEV:

        cc = 3;

        break;

    case -EINVAL:

        cc = 2;

        break;

    case 0:

        cc = 0;

        break;

    default:

        cc = 1;

        break;

    }

    setcc(cpu, cc);

}
