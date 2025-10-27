static void init_ppc_proc(PowerPCCPU *cpu)

{

    PowerPCCPUClass *pcc = POWERPC_CPU_GET_CLASS(cpu);

    CPUPPCState *env = &cpu->env;

#if !defined(CONFIG_USER_ONLY)

    int i;



    env->irq_inputs = NULL;

    /* Set all exception vectors to an invalid address */

    for (i = 0; i < POWERPC_EXCP_NB; i++)

        env->excp_vectors[i] = (target_ulong)(-1ULL);

    env->ivor_mask = 0x00000000;

    env->ivpr_mask = 0x00000000;

    /* Default MMU definitions */

    env->nb_BATs = 0;

    env->nb_tlb = 0;

    env->nb_ways = 0;

    env->tlb_type = TLB_NONE;

#endif

    /* Register SPR common to all PowerPC implementations */

    gen_spr_generic(env);

    spr_register(env, SPR_PVR, "PVR",

                 /* Linux permits userspace to read PVR */

#if defined(CONFIG_LINUX_USER)

                 &spr_read_generic,

#else

                 SPR_NOACCESS,

#endif

                 SPR_NOACCESS,

                 &spr_read_generic, SPR_NOACCESS,

                 pcc->pvr);

    /* Register SVR if it's defined to anything else than POWERPC_SVR_NONE */

    if (pcc->svr != POWERPC_SVR_NONE) {

        if (pcc->svr & POWERPC_SVR_E500) {

            spr_register(env, SPR_E500_SVR, "SVR",

                         SPR_NOACCESS, SPR_NOACCESS,

                         &spr_read_generic, SPR_NOACCESS,

                         pcc->svr & ~POWERPC_SVR_E500);

        } else {

            spr_register(env, SPR_SVR, "SVR",

                         SPR_NOACCESS, SPR_NOACCESS,

                         &spr_read_generic, SPR_NOACCESS,

                         pcc->svr);

        }

    }

    /* PowerPC implementation specific initialisations (SPRs, timers, ...) */

    (*pcc->init_proc)(env);



    /* MSR bits & flags consistency checks */

    if (env->msr_mask & (1 << 25)) {

        switch (env->flags & (POWERPC_FLAG_SPE | POWERPC_FLAG_VRE)) {

        case POWERPC_FLAG_SPE:

        case POWERPC_FLAG_VRE:

            break;

        default:

            fprintf(stderr, "PowerPC MSR definition inconsistency\n"

                    "Should define POWERPC_FLAG_SPE or POWERPC_FLAG_VRE\n");

            exit(1);

        }

    } else if (env->flags & (POWERPC_FLAG_SPE | POWERPC_FLAG_VRE)) {

        fprintf(stderr, "PowerPC MSR definition inconsistency\n"

                "Should not define POWERPC_FLAG_SPE nor POWERPC_FLAG_VRE\n");

        exit(1);

    }

    if (env->msr_mask & (1 << 17)) {

        switch (env->flags & (POWERPC_FLAG_TGPR | POWERPC_FLAG_CE)) {

        case POWERPC_FLAG_TGPR:

        case POWERPC_FLAG_CE:

            break;

        default:

            fprintf(stderr, "PowerPC MSR definition inconsistency\n"

                    "Should define POWERPC_FLAG_TGPR or POWERPC_FLAG_CE\n");

            exit(1);

        }

    } else if (env->flags & (POWERPC_FLAG_TGPR | POWERPC_FLAG_CE)) {

        fprintf(stderr, "PowerPC MSR definition inconsistency\n"

                "Should not define POWERPC_FLAG_TGPR nor POWERPC_FLAG_CE\n");

        exit(1);

    }

    if (env->msr_mask & (1 << 10)) {

        switch (env->flags & (POWERPC_FLAG_SE | POWERPC_FLAG_DWE |

                              POWERPC_FLAG_UBLE)) {

        case POWERPC_FLAG_SE:

        case POWERPC_FLAG_DWE:

        case POWERPC_FLAG_UBLE:

            break;

        default:

            fprintf(stderr, "PowerPC MSR definition inconsistency\n"

                    "Should define POWERPC_FLAG_SE or POWERPC_FLAG_DWE or "

                    "POWERPC_FLAG_UBLE\n");

            exit(1);

        }

    } else if (env->flags & (POWERPC_FLAG_SE | POWERPC_FLAG_DWE |

                             POWERPC_FLAG_UBLE)) {

        fprintf(stderr, "PowerPC MSR definition inconsistency\n"

                "Should not define POWERPC_FLAG_SE nor POWERPC_FLAG_DWE nor "

                "POWERPC_FLAG_UBLE\n");

            exit(1);

    }

    if (env->msr_mask & (1 << 9)) {

        switch (env->flags & (POWERPC_FLAG_BE | POWERPC_FLAG_DE)) {

        case POWERPC_FLAG_BE:

        case POWERPC_FLAG_DE:

            break;

        default:

            fprintf(stderr, "PowerPC MSR definition inconsistency\n"

                    "Should define POWERPC_FLAG_BE or POWERPC_FLAG_DE\n");

            exit(1);

        }

    } else if (env->flags & (POWERPC_FLAG_BE | POWERPC_FLAG_DE)) {

        fprintf(stderr, "PowerPC MSR definition inconsistency\n"

                "Should not define POWERPC_FLAG_BE nor POWERPC_FLAG_DE\n");

        exit(1);

    }

    if (env->msr_mask & (1 << 2)) {

        switch (env->flags & (POWERPC_FLAG_PX | POWERPC_FLAG_PMM)) {

        case POWERPC_FLAG_PX:

        case POWERPC_FLAG_PMM:

            break;

        default:

            fprintf(stderr, "PowerPC MSR definition inconsistency\n"

                    "Should define POWERPC_FLAG_PX or POWERPC_FLAG_PMM\n");

            exit(1);

        }

    } else if (env->flags & (POWERPC_FLAG_PX | POWERPC_FLAG_PMM)) {

        fprintf(stderr, "PowerPC MSR definition inconsistency\n"

                "Should not define POWERPC_FLAG_PX nor POWERPC_FLAG_PMM\n");

        exit(1);

    }

    if ((env->flags & (POWERPC_FLAG_RTC_CLK | POWERPC_FLAG_BUS_CLK)) == 0) {

        fprintf(stderr, "PowerPC flags inconsistency\n"

                "Should define the time-base and decrementer clock source\n");

        exit(1);

    }

    /* Allocate TLBs buffer when needed */

#if !defined(CONFIG_USER_ONLY)

    if (env->nb_tlb != 0) {

        int nb_tlb = env->nb_tlb;

        if (env->id_tlbs != 0)

            nb_tlb *= 2;

        switch (env->tlb_type) {

        case TLB_6XX:

            env->tlb.tlb6 = g_malloc0(nb_tlb * sizeof(ppc6xx_tlb_t));

            break;

        case TLB_EMB:

            env->tlb.tlbe = g_malloc0(nb_tlb * sizeof(ppcemb_tlb_t));

            break;

        case TLB_MAS:

            env->tlb.tlbm = g_malloc0(nb_tlb * sizeof(ppcmas_tlb_t));

            break;

        }

        /* Pre-compute some useful values */

        env->tlb_per_way = env->nb_tlb / env->nb_ways;

    }

    if (env->irq_inputs == NULL) {

        fprintf(stderr, "WARNING: no internal IRQ controller registered.\n"

                " Attempt QEMU to crash very soon !\n");

    }

#endif

    if (env->check_pow == NULL) {

        fprintf(stderr, "WARNING: no power management check handler "

                "registered.\n"

                " Attempt QEMU to crash very soon !\n");

    }

}
