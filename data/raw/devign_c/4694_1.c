static void s390_cpu_realizefn(DeviceState *dev, Error **errp)

{

    S390CPU *cpu = S390_CPU(dev);

    S390CPUClass *scc = S390_CPU_GET_CLASS(dev);



    cpu_reset(CPU(cpu));



    scc->parent_realize(dev, errp);

}
