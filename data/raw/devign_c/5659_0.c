static void xenfv_machine_options(MachineClass *m)

{

    pc_common_machine_options(m);

    m->desc = "Xen Fully-virtualized PC";

    m->max_cpus = HVM_MAX_VCPUS;

    m->default_machine_opts = "accel=xen";

    m->hot_add_cpu = pc_hot_add_cpu;

}
