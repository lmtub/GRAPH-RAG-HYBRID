static int ppc_fixup_cpu(PowerPCCPU *cpu)

{

    CPUPPCState *env = &cpu->env;



    /* TCG doesn't (yet) emulate some groups of instructions that

     * are implemented on some otherwise supported CPUs (e.g. VSX

     * and decimal floating point instructions on POWER7).  We

     * remove unsupported instruction groups from the cpu state's

     * instruction masks and hope the guest can cope.  For at

     * least the pseries machine, the unavailability of these

     * instructions can be advertised to the guest via the device

     * tree. */

    if ((env->insns_flags & ~PPC_TCG_INSNS)

        || (env->insns_flags2 & ~PPC_TCG_INSNS2)) {

        fprintf(stderr, "Warning: Disabling some instructions which are not "

                "emulated by TCG (0x%" PRIx64 ", 0x%" PRIx64 ")\n",

                env->insns_flags & ~PPC_TCG_INSNS,

                env->insns_flags2 & ~PPC_TCG_INSNS2);

    }

    env->insns_flags &= PPC_TCG_INSNS;

    env->insns_flags2 &= PPC_TCG_INSNS2;

    return 0;

}
