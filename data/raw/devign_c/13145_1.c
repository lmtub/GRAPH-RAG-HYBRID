static void pc_q35_2_4_machine_options(MachineClass *m)

{

    PCMachineClass *pcmc = PC_MACHINE_CLASS(m);

    pc_q35_2_5_machine_options(m);

    m->alias = NULL;

    pcmc->broken_reserved_end = true;

    pcmc->inter_dimm_gap = false;

    SET_MACHINE_COMPAT(m, PC_COMPAT_2_4);

}
