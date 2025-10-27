static void ppc_cpu_initfn(Object *obj)

{

    CPUState *cs = CPU(obj);

    PowerPCCPU *cpu = POWERPC_CPU(obj);

    PowerPCCPUClass *pcc = POWERPC_CPU_GET_CLASS(cpu);

    CPUPPCState *env = &cpu->env;



    cs->env_ptr = env;

    cpu_exec_init(env, &error_abort);

    cpu->cpu_dt_id = cs->cpu_index;



    env->msr_mask = pcc->msr_mask;

    env->mmu_model = pcc->mmu_model;

    env->excp_model = pcc->excp_model;

    env->bus_model = pcc->bus_model;

    env->insns_flags = pcc->insns_flags;

    env->insns_flags2 = pcc->insns_flags2;

    env->flags = pcc->flags;

    env->bfd_mach = pcc->bfd_mach;

    env->check_pow = pcc->check_pow;



#if defined(TARGET_PPC64)

    if (pcc->sps) {

        env->sps = *pcc->sps;

    } else if (env->mmu_model & POWERPC_MMU_64) {

        /* Use default sets of page sizes */

        static const struct ppc_segment_page_sizes defsps = {

            .sps = {

                { .page_shift = 12, /* 4K */

                  .slb_enc = 0,

                  .enc = { { .page_shift = 12, .pte_enc = 0 } }

                },

                { .page_shift = 24, /* 16M */

                  .slb_enc = 0x100,

                  .enc = { { .page_shift = 24, .pte_enc = 0 } }

                },

            },

        };

        env->sps = defsps;

    }

#endif /* defined(TARGET_PPC64) */



    if (tcg_enabled()) {

        ppc_translate_init();

    }

}
