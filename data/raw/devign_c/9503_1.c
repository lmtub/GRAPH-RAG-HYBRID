static void create_gic(const VirtBoardInfo *vbi, qemu_irq *pic)

{

    /* We create a standalone GIC v2 */

    DeviceState *gicdev;

    SysBusDevice *gicbusdev;

    const char *gictype = "arm_gic";

    int i;



    if (kvm_irqchip_in_kernel()) {

        gictype = "kvm-arm-gic";

    }



    gicdev = qdev_create(NULL, gictype);

    qdev_prop_set_uint32(gicdev, "revision", 2);

    qdev_prop_set_uint32(gicdev, "num-cpu", smp_cpus);

    /* Note that the num-irq property counts both internal and external

     * interrupts; there are always 32 of the former (mandated by GIC spec).

     */

    qdev_prop_set_uint32(gicdev, "num-irq", NUM_IRQS + 32);

    qdev_init_nofail(gicdev);

    gicbusdev = SYS_BUS_DEVICE(gicdev);

    sysbus_mmio_map(gicbusdev, 0, vbi->memmap[VIRT_GIC_DIST].base);

    sysbus_mmio_map(gicbusdev, 1, vbi->memmap[VIRT_GIC_CPU].base);



    /* Wire the outputs from each CPU's generic timer to the

     * appropriate GIC PPI inputs, and the GIC's IRQ output to

     * the CPU's IRQ input.

     */

    for (i = 0; i < smp_cpus; i++) {

        DeviceState *cpudev = DEVICE(qemu_get_cpu(i));

        int ppibase = NUM_IRQS + i * 32;

        /* physical timer; we wire it up to the non-secure timer's ID,

         * since a real A15 always has TrustZone but QEMU doesn't.

         */

        qdev_connect_gpio_out(cpudev, 0,

                              qdev_get_gpio_in(gicdev, ppibase + 30));

        /* virtual timer */

        qdev_connect_gpio_out(cpudev, 1,

                              qdev_get_gpio_in(gicdev, ppibase + 27));



        sysbus_connect_irq(gicbusdev, i, qdev_get_gpio_in(cpudev, ARM_CPU_IRQ));

    }



    for (i = 0; i < NUM_IRQS; i++) {

        pic[i] = qdev_get_gpio_in(gicdev, i);

    }



    fdt_add_gic_node(vbi);

}
