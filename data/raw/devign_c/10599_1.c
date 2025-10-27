static void spapr_cpu_core_realize_child(Object *child, Error **errp)

{

    Error *local_err = NULL;

    sPAPRMachineState *spapr = SPAPR_MACHINE(qdev_get_machine());

    CPUState *cs = CPU(child);

    PowerPCCPU *cpu = POWERPC_CPU(cs);

    Object *obj;



    obj = object_new(spapr->icp_type);

    object_property_add_child(OBJECT(cpu), "icp", obj, NULL);

    object_property_add_const_link(obj, "xics", OBJECT(spapr), &error_abort);

    object_property_set_bool(obj, true, "realized", &local_err);

    if (local_err) {

        goto error;

    }



    object_property_set_bool(child, true, "realized", &local_err);

    if (local_err) {

        goto error;

    }



    spapr_cpu_init(spapr, cpu, &local_err);

    if (local_err) {

        goto error;

    }



    xics_cpu_setup(XICS_FABRIC(spapr), cpu, ICP(obj));

    return;



error:

    object_unparent(obj);

    error_propagate(errp, local_err);

}
