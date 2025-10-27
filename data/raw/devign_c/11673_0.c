int ioinst_handle_tpi(S390CPU *cpu, uint32_t ipb)

{

    CPUS390XState *env = &cpu->env;

    uint64_t addr;

    int lowcore;

    IOIntCode int_code;

    hwaddr len;

    int ret;

    uint8_t ar;



    trace_ioinst("tpi");

    addr = decode_basedisp_s(env, ipb, &ar);

    if (addr & 3) {

        program_interrupt(env, PGM_SPECIFICATION, 2);

        return -EIO;

    }



    lowcore = addr ? 0 : 1;

    len = lowcore ? 8 /* two words */ : 12 /* three words */;

    ret = css_do_tpi(&int_code, lowcore);

    if (ret == 1) {

        s390_cpu_virt_mem_write(cpu, lowcore ? 184 : addr, ar, &int_code, len);

    }

    return ret;

}
