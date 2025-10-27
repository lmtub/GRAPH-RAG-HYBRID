static bool machine_get_kernel_irqchip(Object *obj, Error **errp)

{

    MachineState *ms = MACHINE(obj);



    return ms->kernel_irqchip;

}
