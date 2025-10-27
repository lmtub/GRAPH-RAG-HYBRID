static int spapr_cpu_core_realize_child(Object *child, void *opaque)

{

    Error **errp = opaque, *local_err = NULL;

    sPAPRMachineState *spapr = SPAPR_MACHINE(qdev_get_machine());

    CPUState *cs = CPU(child);

    PowerPCCPU *cpu = POWERPC_CPU(cs);



    object_property_set_bool(child, true, "realized", &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        return 1;

    }



    spapr_cpu_init(spapr, cpu, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        return 1;

    }

    return 0;

}
