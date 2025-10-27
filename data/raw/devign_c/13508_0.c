static void spapr_cpu_init(sPAPRMachineState *spapr, PowerPCCPU *cpu,

                           Error **errp)

{

    CPUPPCState *env = &cpu->env;



    /* Set time-base frequency to 512 MHz */

    cpu_ppc_tb_init(env, SPAPR_TIMEBASE_FREQ);



    /* Enable PAPR mode in TCG or KVM */

    cpu_ppc_set_papr(cpu, PPC_VIRTUAL_HYPERVISOR(spapr));



    if (spapr->max_compat_pvr) {

        Error *local_err = NULL;



        ppc_set_compat(cpu, spapr->max_compat_pvr, &local_err);

        if (local_err) {

            error_propagate(errp, local_err);

            return;

        }

    }



    qemu_register_reset(spapr_cpu_reset, cpu);

    spapr_cpu_reset(cpu);

}
