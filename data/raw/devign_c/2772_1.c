static void rtas_stop_self(PowerPCCPU *cpu, sPAPRMachineState *spapr,

                           uint32_t token, uint32_t nargs,

                           target_ulong args,

                           uint32_t nret, target_ulong rets)

{

    CPUState *cs = CPU(cpu);

    CPUPPCState *env = &cpu->env;




    cs->halted = 1;

    qemu_cpu_kick(cs);

    /*

     * While stopping a CPU, the guest calls H_CPPR which

     * effectively disables interrupts on XICS level.

     * However decrementer interrupts in TCG can still

     * wake the CPU up so here we disable interrupts in MSR

     * as well.

     * As rtas_start_cpu() resets the whole MSR anyway, there is

     * no need to bother with specific bits, we just clear it.

     */

    env->msr = 0;



    /* Disable Power-saving mode Exit Cause exceptions for the CPU.

     * This could deliver an interrupt on a dying CPU and crash the

     * guest */

    env->spr[SPR_LPCR] &= ~pcc->lpcr_pm;

}