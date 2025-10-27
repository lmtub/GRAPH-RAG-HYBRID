static target_ulong h_set_mode_resouce_addr_trans_mode(PowerPCCPU *cpu,

                                                       target_ulong mflags,

                                                       target_ulong value1,

                                                       target_ulong value2)

{

    CPUState *cs;

    PowerPCCPUClass *pcc = POWERPC_CPU_GET_CLASS(cpu);

    target_ulong prefix;



    if (!(pcc->insns_flags2 & PPC2_ISA207S)) {

        return H_P2;

    }

    if (value1) {

        return H_P3;

    }

    if (value2) {

        return H_P4;

    }



    switch (mflags) {

    case H_SET_MODE_ADDR_TRANS_NONE:

        prefix = 0;

        break;

    case H_SET_MODE_ADDR_TRANS_0001_8000:

        prefix = 0x18000;

        break;

    case H_SET_MODE_ADDR_TRANS_C000_0000_0000_4000:

        prefix = 0xC000000000004000;

        break;

    default:

        return H_UNSUPPORTED_FLAG;

    }



    CPU_FOREACH(cs) {

        CPUPPCState *env = &POWERPC_CPU(cpu)->env;



        set_spr(cs, SPR_LPCR, mflags << LPCR_AIL_SHIFT, LPCR_AIL);

        env->excp_prefix = prefix;

    }



    return H_SUCCESS;

}
