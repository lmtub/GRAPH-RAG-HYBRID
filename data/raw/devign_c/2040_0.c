static void mvp_init (CPUMIPSState *env, const mips_def_t *def)

{

    env->mvp = qemu_mallocz(sizeof(CPUMIPSMVPContext));



    /* MVPConf1 implemented, TLB sharable, no gating storage support,

       programmable cache partitioning implemented, number of allocatable

       and sharable TLB entries, MVP has allocatable TCs, 2 VPEs

       implemented, 5 TCs implemented. */

    env->mvp->CP0_MVPConf0 = (1 << CP0MVPC0_M) | (1 << CP0MVPC0_TLBS) |

                             (0 << CP0MVPC0_GS) | (1 << CP0MVPC0_PCP) |

// TODO: actually do 2 VPEs.

//                             (1 << CP0MVPC0_TCA) | (0x1 << CP0MVPC0_PVPE) |

//                             (0x04 << CP0MVPC0_PTC);

                             (1 << CP0MVPC0_TCA) | (0x0 << CP0MVPC0_PVPE) |

                             (0x04 << CP0MVPC0_PTC);

    /* Usermode has no TLB support */

    if (!env->user_mode_only)

        env->mvp->CP0_MVPConf0 |= (env->tlb->nb_tlb << CP0MVPC0_PTLBE);



    /* Allocatable CP1 have media extensions, allocatable CP1 have FP support,

       no UDI implemented, no CP2 implemented, 1 CP1 implemented. */

    env->mvp->CP0_MVPConf1 = (1 << CP0MVPC1_CIM) | (1 << CP0MVPC1_CIF) |

                             (0x0 << CP0MVPC1_PCX) | (0x0 << CP0MVPC1_PCP2) |

                             (0x1 << CP0MVPC1_PCP1);

}
