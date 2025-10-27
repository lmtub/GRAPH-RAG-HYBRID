static qemu_irq *ppce500_init_mpic(PPCE500Params *params, MemoryRegion *ccsr,

                                   qemu_irq **irqs)

{

    qemu_irq *mpic;

    DeviceState *dev;

    SysBusDevice *s;

    int i, j, k;



    mpic = g_new(qemu_irq, 256);

    dev = qdev_create(NULL, "openpic");

    qdev_prop_set_uint32(dev, "nb_cpus", smp_cpus);

    qdev_prop_set_uint32(dev, "model", params->mpic_version);

    qdev_init_nofail(dev);

    s = SYS_BUS_DEVICE(dev);



    k = 0;

    for (i = 0; i < smp_cpus; i++) {

        for (j = 0; j < OPENPIC_OUTPUT_NB; j++) {

            sysbus_connect_irq(s, k++, irqs[i][j]);

        }

    }



    for (i = 0; i < 256; i++) {

        mpic[i] = qdev_get_gpio_in(dev, i);

    }



    memory_region_add_subregion(ccsr, MPC8544_MPIC_REGS_OFFSET,

                                s->mmio[0].memory);



    return mpic;

}
