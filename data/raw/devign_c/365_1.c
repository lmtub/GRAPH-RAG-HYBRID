static void mips_cpu_realizefn(DeviceState *dev, Error **errp)

{

    MIPSCPU *cpu = MIPS_CPU(dev);

    MIPSCPUClass *mcc = MIPS_CPU_GET_CLASS(dev);



    cpu_reset(CPU(cpu));



    mcc->parent_realize(dev, errp);

}
