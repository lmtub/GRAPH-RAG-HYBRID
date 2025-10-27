static void kvm_arm_gic_realize(DeviceState *dev, Error **errp)

{

    int i;

    GICState *s = KVM_ARM_GIC(dev);

    SysBusDevice *sbd = SYS_BUS_DEVICE(dev);

    KVMARMGICClass *kgc = KVM_ARM_GIC_GET_CLASS(s);



    kgc->parent_realize(dev, errp);

    if (error_is_set(errp)) {

        return;

    }



    i = s->num_irq - GIC_INTERNAL;

    /* For the GIC, also expose incoming GPIO lines for PPIs for each CPU.

     * GPIO array layout is thus:

     *  [0..N-1] SPIs

     *  [N..N+31] PPIs for CPU 0

     *  [N+32..N+63] PPIs for CPU 1

     *   ...

     */

    i += (GIC_INTERNAL * s->num_cpu);

    qdev_init_gpio_in(dev, kvm_arm_gic_set_irq, i);

    /* We never use our outbound IRQ lines but provide them so that

     * we maintain the same interface as the non-KVM GIC.

     */

    for (i = 0; i < s->num_cpu; i++) {

        sysbus_init_irq(sbd, &s->parent_irq[i]);

    }

    /* Distributor */

    memory_region_init_reservation(&s->iomem, OBJECT(s),

                                   "kvm-gic_dist", 0x1000);

    sysbus_init_mmio(sbd, &s->iomem);

    kvm_arm_register_device(&s->iomem,

                            (KVM_ARM_DEVICE_VGIC_V2 << KVM_ARM_DEVICE_ID_SHIFT)

                            | KVM_VGIC_V2_ADDR_TYPE_DIST);

    /* CPU interface for current core. Unlike arm_gic, we don't

     * provide the "interface for core #N" memory regions, because

     * cores with a VGIC don't have those.

     */

    memory_region_init_reservation(&s->cpuiomem[0], OBJECT(s),

                                   "kvm-gic_cpu", 0x1000);

    sysbus_init_mmio(sbd, &s->cpuiomem[0]);

    kvm_arm_register_device(&s->cpuiomem[0],

                            (KVM_ARM_DEVICE_VGIC_V2 << KVM_ARM_DEVICE_ID_SHIFT)

                            | KVM_VGIC_V2_ADDR_TYPE_CPU);

}
