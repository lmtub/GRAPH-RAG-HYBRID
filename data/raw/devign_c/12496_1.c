static uint32_t get_elf_hwcap(void)

{

    CPUState *e = thread_env;

    uint32_t features = 0;



    /* We don't have to be terribly complete here; the high points are

       Altivec/FP/SPE support.  Anything else is just a bonus.  */

#define GET_FEATURE(flag, feature)              \

    do {if (e->insns_flags & flag) features |= feature; } while(0)

    GET_FEATURE(PPC_64B, PPC_FEATURE_64);

    GET_FEATURE(PPC_FLOAT, PPC_FEATURE_HAS_FPU);

    GET_FEATURE(PPC_ALTIVEC, PPC_FEATURE_HAS_ALTIVEC);

    GET_FEATURE(PPC_SPE, PPC_FEATURE_HAS_SPE);

    GET_FEATURE(PPC_SPE_SINGLE, PPC_FEATURE_HAS_EFP_SINGLE);

    GET_FEATURE(PPC_SPE_DOUBLE, PPC_FEATURE_HAS_EFP_DOUBLE);

    GET_FEATURE(PPC_BOOKE, PPC_FEATURE_BOOKE);

    GET_FEATURE(PPC_405_MAC, PPC_FEATURE_HAS_4xxMAC);

#undef GET_FEATURE



    return features;

}
