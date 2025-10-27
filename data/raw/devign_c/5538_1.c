static void icp_realize(DeviceState *dev, Error **errp)

{

    ICPState *icp = ICP(dev);

    ICPStateClass *icpc = ICP_GET_CLASS(dev);

    PowerPCCPU *cpu;

    CPUPPCState *env;

    Object *obj;

    Error *err = NULL;



    obj = object_property_get_link(OBJECT(dev), ICP_PROP_XICS, &err);

    if (!obj) {

        error_setg(errp, "%s: required link '" ICP_PROP_XICS "' not found: %s",

                   __func__, error_get_pretty(err));

        return;

    }



    icp->xics = XICS_FABRIC(obj);



    obj = object_property_get_link(OBJECT(dev), ICP_PROP_CPU, &err);

    if (!obj) {

        error_setg(errp, "%s: required link '" ICP_PROP_CPU "' not found: %s",

                   __func__, error_get_pretty(err));

        return;

    }



    cpu = POWERPC_CPU(obj);

    cpu->intc = OBJECT(icp);

    icp->cs = CPU(obj);



    env = &cpu->env;

    switch (PPC_INPUT(env)) {

    case PPC_FLAGS_INPUT_POWER7:

        icp->output = env->irq_inputs[POWER7_INPUT_INT];

        break;



    case PPC_FLAGS_INPUT_970:

        icp->output = env->irq_inputs[PPC970_INPUT_INT];

        break;



    default:

        error_setg(errp, "XICS interrupt controller does not support this CPU bus model");

        return;

    }



    if (icpc->realize) {

        icpc->realize(icp, errp);

    }



    qemu_register_reset(icp_reset, dev);

    vmstate_register(NULL, icp->cs->cpu_index, &vmstate_icp_server, icp);

}
