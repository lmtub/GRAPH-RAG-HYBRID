static void openrisc_cpu_realizefn(DeviceState *dev, Error **errp)

{

    OpenRISCCPU *cpu = OPENRISC_CPU(dev);

    OpenRISCCPUClass *occ = OPENRISC_CPU_GET_CLASS(dev);



    cpu_reset(CPU(cpu));



    occ->parent_realize(dev, errp);

}
