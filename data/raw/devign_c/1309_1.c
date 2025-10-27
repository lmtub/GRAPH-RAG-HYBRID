void ppc_cpu_dump_state(CPUState *cs, FILE *f, fprintf_function cpu_fprintf,

                        int flags)

{

#define RGPL  4

#define RFPL  4



    PowerPCCPU *cpu = POWERPC_CPU(cs);

    CPUPPCState *env = &cpu->env;

    int i;



    cpu_fprintf(f, "NIP " TARGET_FMT_lx "   LR " TARGET_FMT_lx " CTR "

                TARGET_FMT_lx " XER " TARGET_FMT_lx " CPU#%d\n",

                env->nip, env->lr, env->ctr, cpu_read_xer(env),

                cs->cpu_index);

    cpu_fprintf(f, "MSR " TARGET_FMT_lx " HID0 " TARGET_FMT_lx "  HF "

                TARGET_FMT_lx " iidx %d didx %d\n",

                env->msr, env->spr[SPR_HID0],

                env->hflags, env->immu_idx, env->dmmu_idx);

#if !defined(NO_TIMER_DUMP)

    cpu_fprintf(f, "TB %08" PRIu32 " %08" PRIu64

#if !defined(CONFIG_USER_ONLY)

                " DECR %08" PRIu32


                "\n",

                cpu_ppc_load_tbu(env), cpu_ppc_load_tbl(env)

#if !defined(CONFIG_USER_ONLY)

                , cpu_ppc_load_decr(env)


                );


    for (i = 0; i < 32; i++) {

        if ((i & (RGPL - 1)) == 0)

            cpu_fprintf(f, "GPR%02d", i);

        cpu_fprintf(f, " %016" PRIx64, ppc_dump_gpr(env, i));

        if ((i & (RGPL - 1)) == (RGPL - 1))

            cpu_fprintf(f, "\n");


    cpu_fprintf(f, "CR ");

    for (i = 0; i < 8; i++)

        cpu_fprintf(f, "%01x", env->crf[i]);

    cpu_fprintf(f, "  [");

    for (i = 0; i < 8; i++) {

        char a = '-';

        if (env->crf[i] & 0x08)

            a = 'L';

        else if (env->crf[i] & 0x04)

            a = 'G';

        else if (env->crf[i] & 0x02)

            a = 'E';

        cpu_fprintf(f, " %c%c", a, env->crf[i] & 0x01 ? 'O' : ' ');


    cpu_fprintf(f, " ]             RES " TARGET_FMT_lx "\n",

                env->reserve_addr);

    for (i = 0; i < 32; i++) {

        if ((i & (RFPL - 1)) == 0)

            cpu_fprintf(f, "FPR%02d", i);

        cpu_fprintf(f, " %016" PRIx64, *((uint64_t *)&env->fpr[i]));

        if ((i & (RFPL - 1)) == (RFPL - 1))

            cpu_fprintf(f, "\n");


    cpu_fprintf(f, "FPSCR " TARGET_FMT_lx "\n", env->fpscr);

#if !defined(CONFIG_USER_ONLY)

    cpu_fprintf(f, " SRR0 " TARGET_FMT_lx "  SRR1 " TARGET_FMT_lx

                   "    PVR " TARGET_FMT_lx " VRSAVE " TARGET_FMT_lx "\n",

                env->spr[SPR_SRR0], env->spr[SPR_SRR1],

                env->spr[SPR_PVR], env->spr[SPR_VRSAVE]);



    cpu_fprintf(f, "SPRG0 " TARGET_FMT_lx " SPRG1 " TARGET_FMT_lx

                   "  SPRG2 " TARGET_FMT_lx "  SPRG3 " TARGET_FMT_lx "\n",

                env->spr[SPR_SPRG0], env->spr[SPR_SPRG1],

                env->spr[SPR_SPRG2], env->spr[SPR_SPRG3]);



    cpu_fprintf(f, "SPRG4 " TARGET_FMT_lx " SPRG5 " TARGET_FMT_lx

                   "  SPRG6 " TARGET_FMT_lx "  SPRG7 " TARGET_FMT_lx "\n",

                env->spr[SPR_SPRG4], env->spr[SPR_SPRG5],

                env->spr[SPR_SPRG6], env->spr[SPR_SPRG7]);










    if (env->excp_model == POWERPC_EXCP_BOOKE) {

        cpu_fprintf(f, "CSRR0 " TARGET_FMT_lx " CSRR1 " TARGET_FMT_lx

                       " MCSRR0 " TARGET_FMT_lx " MCSRR1 " TARGET_FMT_lx "\n",

                    env->spr[SPR_BOOKE_CSRR0], env->spr[SPR_BOOKE_CSRR1],

                    env->spr[SPR_BOOKE_MCSRR0], env->spr[SPR_BOOKE_MCSRR1]);



        cpu_fprintf(f, "  TCR " TARGET_FMT_lx "   TSR " TARGET_FMT_lx

                       "    ESR " TARGET_FMT_lx "   DEAR " TARGET_FMT_lx "\n",

                    env->spr[SPR_BOOKE_TCR], env->spr[SPR_BOOKE_TSR],

                    env->spr[SPR_BOOKE_ESR], env->spr[SPR_BOOKE_DEAR]);



        cpu_fprintf(f, "  PIR " TARGET_FMT_lx " DECAR " TARGET_FMT_lx

                       "   IVPR " TARGET_FMT_lx "   EPCR " TARGET_FMT_lx "\n",

                    env->spr[SPR_BOOKE_PIR], env->spr[SPR_BOOKE_DECAR],

                    env->spr[SPR_BOOKE_IVPR], env->spr[SPR_BOOKE_EPCR]);



        cpu_fprintf(f, " MCSR " TARGET_FMT_lx " SPRG8 " TARGET_FMT_lx

                       "    EPR " TARGET_FMT_lx "\n",

                    env->spr[SPR_BOOKE_MCSR], env->spr[SPR_BOOKE_SPRG8],

                    env->spr[SPR_BOOKE_EPR]);



        /* FSL-specific */

        cpu_fprintf(f, " MCAR " TARGET_FMT_lx "  PID1 " TARGET_FMT_lx

                       "   PID2 " TARGET_FMT_lx "    SVR " TARGET_FMT_lx "\n",

                    env->spr[SPR_Exxx_MCAR], env->spr[SPR_BOOKE_PID1],

                    env->spr[SPR_BOOKE_PID2], env->spr[SPR_E500_SVR]);



        /*

         * IVORs are left out as they are large and do not change often --

         * they can be read with "p $ivor0", "p $ivor1", etc.

         */





    if (env->flags & POWERPC_FLAG_CFAR) {

        cpu_fprintf(f, " CFAR " TARGET_FMT_lx"\n", env->cfar);





    switch (env->mmu_model) {

    case POWERPC_MMU_32B:

    case POWERPC_MMU_601:

    case POWERPC_MMU_SOFT_6xx:

    case POWERPC_MMU_SOFT_74xx:


    case POWERPC_MMU_64B:

    case POWERPC_MMU_2_03:

    case POWERPC_MMU_2_06:

    case POWERPC_MMU_2_06a:

    case POWERPC_MMU_2_07:

    case POWERPC_MMU_2_07a:


        cpu_fprintf(f, " SDR1 " TARGET_FMT_lx "   DAR " TARGET_FMT_lx

                       "  DSISR " TARGET_FMT_lx "\n", env->spr[SPR_SDR1],

                    env->spr[SPR_DAR], env->spr[SPR_DSISR]);

        break;

    case POWERPC_MMU_BOOKE206:

        cpu_fprintf(f, " MAS0 " TARGET_FMT_lx "  MAS1 " TARGET_FMT_lx

                       "   MAS2 " TARGET_FMT_lx "   MAS3 " TARGET_FMT_lx "\n",

                    env->spr[SPR_BOOKE_MAS0], env->spr[SPR_BOOKE_MAS1],

                    env->spr[SPR_BOOKE_MAS2], env->spr[SPR_BOOKE_MAS3]);



        cpu_fprintf(f, " MAS4 " TARGET_FMT_lx "  MAS6 " TARGET_FMT_lx

                       "   MAS7 " TARGET_FMT_lx "    PID " TARGET_FMT_lx "\n",

                    env->spr[SPR_BOOKE_MAS4], env->spr[SPR_BOOKE_MAS6],

                    env->spr[SPR_BOOKE_MAS7], env->spr[SPR_BOOKE_PID]);



        cpu_fprintf(f, "MMUCFG " TARGET_FMT_lx " TLB0CFG " TARGET_FMT_lx

                       " TLB1CFG " TARGET_FMT_lx "\n",

                    env->spr[SPR_MMUCFG], env->spr[SPR_BOOKE_TLB0CFG],

                    env->spr[SPR_BOOKE_TLB1CFG]);

        break;

    default:

        break;





#undef RGPL

#undef RFPL
