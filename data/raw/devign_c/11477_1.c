static void xtensa_cpu_realizefn(DeviceState *dev, Error **errp)
{
    CPUState *cs = CPU(dev);
    XtensaCPUClass *xcc = XTENSA_CPU_GET_CLASS(dev);
    cs->gdb_num_regs = xcc->config->gdb_regmap.num_regs;
    xcc->parent_realize(dev, errp);
}