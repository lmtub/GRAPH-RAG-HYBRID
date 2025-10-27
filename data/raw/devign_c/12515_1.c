static void spapr_cpu_reset(void *opaque)

{

    PowerPCCPU *cpu = opaque;

    CPUState *cs = CPU(cpu);

    CPUPPCState *env = &cpu->env;



    cpu_reset(cs);



    /* All CPUs start halted.  CPU0 is unhalted from the machine level

     * reset code and the rest are explicitly started up by the guest

     * using an RTAS call */

    cs->halted = 1;



    env->spr[SPR_HIOR] = 0;



    env->external_htab = (uint8_t *)spapr->htab;

    if (kvm_enabled() && !env->external_htab) {

        /*

         * HV KVM, set external_htab to 1 so our ppc_hash64_load_hpte*

         * functions do the right thing.

         */

        env->external_htab = (void *)1;

    }

    env->htab_base = -1;

    env->htab_mask = HTAB_SIZE(spapr) - 1;

    env->spr[SPR_SDR1] = (target_ulong)(uintptr_t)spapr->htab |

        (spapr->htab_shift - 18);

}
