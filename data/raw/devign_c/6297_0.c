int cpu_ppc_register (CPUPPCState *env, ppc_def_t *def)

{

    env->msr_mask = def->msr_mask;

    env->mmu_model = def->mmu_model;

    env->excp_model = def->excp_model;

    env->bus_model = def->bus_model;

    env->bfd_mach = def->bfd_mach;

    if (create_ppc_opcodes(env, def) < 0)

        return -1;

    init_ppc_proc(env, def);

#if defined(PPC_DUMP_CPU)

    {

        const unsigned char *mmu_model, *excp_model, *bus_model;

        switch (env->mmu_model) {

        case POWERPC_MMU_32B:

            mmu_model = "PowerPC 32";

            break;

        case POWERPC_MMU_64B:

            mmu_model = "PowerPC 64";

            break;

        case POWERPC_MMU_601:

            mmu_model = "PowerPC 601";

            break;

        case POWERPC_MMU_SOFT_6xx:

            mmu_model = "PowerPC 6xx/7xx with software driven TLBs";

            break;

        case POWERPC_MMU_SOFT_74xx:

            mmu_model = "PowerPC 74xx with software driven TLBs";

            break;

        case POWERPC_MMU_SOFT_4xx:

            mmu_model = "PowerPC 4xx with software driven TLBs";

            break;

        case POWERPC_MMU_SOFT_4xx_Z:

            mmu_model = "PowerPC 4xx with software driven TLBs "

                "and zones protections";

            break;

        case POWERPC_MMU_REAL_4xx:

            mmu_model = "PowerPC 4xx real mode only";

            break;

        case POWERPC_MMU_BOOKE:

            mmu_model = "PowerPC BookE";

            break;

        case POWERPC_MMU_BOOKE_FSL:

            mmu_model = "PowerPC BookE FSL";

            break;

        case POWERPC_MMU_64BRIDGE:

            mmu_model = "PowerPC 64 bridge";

            break;

        default:

            mmu_model = "Unknown or invalid";

            break;

        }

        switch (env->excp_model) {

        case POWERPC_EXCP_STD:

            excp_model = "PowerPC";

            break;

        case POWERPC_EXCP_40x:

            excp_model = "PowerPC 40x";

            break;

        case POWERPC_EXCP_601:

            excp_model = "PowerPC 601";

            break;

        case POWERPC_EXCP_602:

            excp_model = "PowerPC 602";

            break;

        case POWERPC_EXCP_603:

            excp_model = "PowerPC 603";

            break;

        case POWERPC_EXCP_603E:

            excp_model = "PowerPC 603e";

            break;

        case POWERPC_EXCP_604:

            excp_model = "PowerPC 604";

            break;

        case POWERPC_EXCP_7x0:

            excp_model = "PowerPC 740/750";

            break;

        case POWERPC_EXCP_7x5:

            excp_model = "PowerPC 745/755";

            break;

        case POWERPC_EXCP_74xx:

            excp_model = "PowerPC 74xx";

            break;

        case POWERPC_EXCP_970:

            excp_model = "PowerPC 970";

            break;

        case POWERPC_EXCP_BOOKE:

            excp_model = "PowerPC BookE";

            break;

        default:

            excp_model = "Unknown or invalid";

            break;

        }

        switch (env->bus_model) {

        case PPC_FLAGS_INPUT_6xx:

            bus_model = "PowerPC 6xx";

            break;

        case PPC_FLAGS_INPUT_BookE:

            bus_model = "PowerPC BookE";

            break;

        case PPC_FLAGS_INPUT_405:

            bus_model = "PowerPC 405";

            break;

        case PPC_FLAGS_INPUT_970:

            bus_model = "PowerPC 970";

            break;

        case PPC_FLAGS_INPUT_401:

            bus_model = "PowerPC 401/403";

            break;

        default:

            bus_model = "Unknown or invalid";

            break;

        }

        printf("PowerPC %-12s : PVR %08x MSR %016" PRIx64 "\n"

               "    MMU model        : %s\n",

               def->name, def->pvr, def->msr_mask, mmu_model);

        if (env->tlb != NULL) {

            printf("                       %d %s TLB in %d ways\n",

                   env->nb_tlb, env->id_tlbs ? "splitted" : "merged",

                   env->nb_ways);

        }

        printf("    Exceptions model : %s\n"

               "    Bus model        : %s\n",

               excp_model, bus_model);

    }

    dump_ppc_insns(env);

    dump_ppc_sprs(env);

    fflush(stdout);

#endif



    return 0;

}
