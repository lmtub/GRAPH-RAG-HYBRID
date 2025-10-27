static void emulate_spapr_hypercall(PPCVirtualHypervisor *vhyp,
                                    PowerPCCPU *cpu)
{
    CPUPPCState *env = &cpu->env;
    if (msr_pr) {
        hcall_dprintf("Hypercall made with MSR[PR]=1\n");
        env->gpr[3] = H_PRIVILEGE;
    } else {
        env->gpr[3] = spapr_hypercall(cpu, env->gpr[3], &env->gpr[4]);
    }
}