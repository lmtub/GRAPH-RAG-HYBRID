static void uc32_cpu_realizefn(DeviceState *dev, Error **errp)
{
    UniCore32CPUClass *ucc = UNICORE32_CPU_GET_CLASS(dev);
    ucc->parent_realize(dev, errp);
}