static qemu_irq *ppce500_init_mpic(MachineState *machine, PPCE500Params *params,

                                   MemoryRegion *ccsr, qemu_irq **irqs)

{

    qemu_irq *mpic;

    DeviceState *dev = NULL;

    SysBusDevice *s;

    int i;



    mpic = g_new0(qemu_irq, 256);



    if (kvm_enabled()) {

        Error *err = NULL;



        if (machine_kernel_irqchip_allowed(machine)) {

            dev = ppce500_init_mpic_kvm(params, irqs, &err);

        }

        if (machine_kernel_irqchip_required(machine) && !dev) {

            error_reportf_err(err,

                              "kernel_irqchip requested but unavailable: ");

            exit(1);

        }

    }



    if (!dev) {

        dev = ppce500_init_mpic_qemu(params, irqs);

    }



    for (i = 0; i < 256; i++) {

        mpic[i] = qdev_get_gpio_in(dev, i);

    }



    s = SYS_BUS_DEVICE(dev);

    memory_region_add_subregion(ccsr, MPC8544_MPIC_REGS_OFFSET,

                                s->mmio[0].memory);



    return mpic;

}
