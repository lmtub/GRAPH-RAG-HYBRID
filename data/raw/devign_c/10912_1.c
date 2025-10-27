static void kvm_arm_gic_realize(DeviceState *dev, Error **errp)
{
    int i;
    GICState *s = KVM_ARM_GIC(dev);
    SysBusDevice *sbd = SYS_BUS_DEVICE(dev);
    KVMARMGICClass *kgc = KVM_ARM_GIC_GET_CLASS(s);
    Error *local_err = NULL;
    int ret;
    kgc->parent_realize(dev, &local_err);
    if (local_err) {
        error_propagate(errp, local_err);
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
    /* We never use our outbound IRQ/FIQ lines but provide them so that
     * we maintain the same interface as the non-KVM GIC.
     */
    for (i = 0; i < s->num_cpu; i++) {
        sysbus_init_irq(sbd, &s->parent_irq[i]);
    for (i = 0; i < s->num_cpu; i++) {
        sysbus_init_irq(sbd, &s->parent_fiq[i]);
    /* Try to create the device via the device control API */
    s->dev_fd = -1;
    ret = kvm_create_device(kvm_state, KVM_DEV_TYPE_ARM_VGIC_V2, false);
    if (ret >= 0) {
        s->dev_fd = ret;
    } else if (ret != -ENODEV && ret != -ENOTSUP) {
        error_setg_errno(errp, -ret, "error creating in-kernel VGIC");
    if (kvm_gic_supports_attr(s, KVM_DEV_ARM_VGIC_GRP_NR_IRQS, 0)) {
        uint32_t numirqs = s->num_irq;
        kvm_gic_access(s, KVM_DEV_ARM_VGIC_GRP_NR_IRQS, 0, 0, &numirqs, 1);
    /* Tell the kernel to complete VGIC initialization now */
    if (kvm_gic_supports_attr(s, KVM_DEV_ARM_VGIC_GRP_CTRL,
                              KVM_DEV_ARM_VGIC_CTRL_INIT)) {
        kvm_gic_access(s, KVM_DEV_ARM_VGIC_GRP_CTRL,
                          KVM_DEV_ARM_VGIC_CTRL_INIT, 0, 0, 1);
    /* Distributor */
    memory_region_init_reservation(&s->iomem, OBJECT(s),
                                   "kvm-gic_dist", 0x1000);
    sysbus_init_mmio(sbd, &s->iomem);
    kvm_arm_register_device(&s->iomem,
                            (KVM_ARM_DEVICE_VGIC_V2 << KVM_ARM_DEVICE_ID_SHIFT)
                            | KVM_VGIC_V2_ADDR_TYPE_DIST,
                            KVM_DEV_ARM_VGIC_GRP_ADDR,
                            KVM_VGIC_V2_ADDR_TYPE_DIST,
                            s->dev_fd);
    /* CPU interface for current core. Unlike arm_gic, we don't
     * provide the "interface for core #N" memory regions, because
     * cores with a VGIC don't have those.
     */
    memory_region_init_reservation(&s->cpuiomem[0], OBJECT(s),
                                   "kvm-gic_cpu", 0x1000);
    sysbus_init_mmio(sbd, &s->cpuiomem[0]);
    kvm_arm_register_device(&s->cpuiomem[0],
                            (KVM_ARM_DEVICE_VGIC_V2 << KVM_ARM_DEVICE_ID_SHIFT)
                            | KVM_VGIC_V2_ADDR_TYPE_CPU,
                            KVM_DEV_ARM_VGIC_GRP_ADDR,
                            KVM_VGIC_V2_ADDR_TYPE_CPU,
                            s->dev_fd);