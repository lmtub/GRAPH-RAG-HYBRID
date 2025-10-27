static void spapr_machine_2_6_class_options(MachineClass *mc)

{

    sPAPRMachineClass *smc = SPAPR_MACHINE_CLASS(mc);



    spapr_machine_2_7_class_options(mc);

    smc->dr_cpu_enabled = false;

    SET_MACHINE_COMPAT(mc, SPAPR_COMPAT_2_6);

}
