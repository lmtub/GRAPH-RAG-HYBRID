static int arm_gic_init(SysBusDevice *dev)

{

    /* Device instance init function for the GIC sysbus device */

    int i;

    GICState *s = FROM_SYSBUS(GICState, dev);

    ARMGICClass *agc = ARM_GIC_GET_CLASS(s);



    agc->parent_init(dev);



    gic_init_irqs_and_distributor(s, s->num_irq);



    /* Memory regions for the CPU interfaces (NVIC doesn't have these):

     * a region for "CPU interface for this core", then a region for

     * "CPU interface for core 0", "for core 1", ...

     * NB that the memory region size of 0x100 applies for the 11MPCore

     * and also cores following the GIC v1 spec (ie A9).

     * GIC v2 defines a larger memory region (0x1000) so this will need

     * to be extended when we implement A15.

     */

    memory_region_init_io(&s->cpuiomem[0], &gic_thiscpu_ops, s,

                          "gic_cpu", 0x100);

    for (i = 0; i < NUM_CPU(s); i++) {

        s->backref[i] = s;

        memory_region_init_io(&s->cpuiomem[i+1], &gic_cpu_ops, &s->backref[i],

                              "gic_cpu", 0x100);

    }

    /* Distributor */

    sysbus_init_mmio(dev, &s->iomem);

    /* cpu interfaces (one for "current cpu" plus one per cpu) */

    for (i = 0; i <= NUM_CPU(s); i++) {

        sysbus_init_mmio(dev, &s->cpuiomem[i]);

    }

    return 0;

}
