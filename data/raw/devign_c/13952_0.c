static void kvm_openpic_realize(DeviceState *dev, Error **errp)

{

    SysBusDevice *d = SYS_BUS_DEVICE(dev);

    KVMOpenPICState *opp = KVM_OPENPIC(dev);

    KVMState *s = kvm_state;

    int kvm_openpic_model;

    struct kvm_create_device cd = {0};

    int ret, i;



    if (!kvm_check_extension(s, KVM_CAP_DEVICE_CTRL)) {

        error_setg(errp, "Kernel is lacking Device Control API");

        return;

    }



    switch (opp->model) {

    case OPENPIC_MODEL_FSL_MPIC_20:

        kvm_openpic_model = KVM_DEV_TYPE_FSL_MPIC_20;

        break;



    case OPENPIC_MODEL_FSL_MPIC_42:

        kvm_openpic_model = KVM_DEV_TYPE_FSL_MPIC_42;

        break;



    default:

        error_setg(errp, "Unsupported OpenPIC model %" PRIu32, opp->model);

        return;

    }



    cd.type = kvm_openpic_model;

    ret = kvm_vm_ioctl(s, KVM_CREATE_DEVICE, &cd);

    if (ret < 0) {

        error_setg(errp, "Can't create device %d: %s",

                   cd.type, strerror(errno));

        return;

    }

    opp->fd = cd.fd;



    sysbus_init_mmio(d, &opp->mem);

    qdev_init_gpio_in(dev, kvm_openpic_set_irq, OPENPIC_MAX_IRQ);



    opp->mem_listener.region_add = kvm_openpic_region_add;

    opp->mem_listener.region_del = kvm_openpic_region_del;

    memory_listener_register(&opp->mem_listener, &address_space_memory);



    /* indicate pic capabilities */

    msi_supported = true;

    kvm_kernel_irqchip = true;

    kvm_async_interrupts_allowed = true;



    /* set up irq routing */

    kvm_init_irq_routing(kvm_state);

    for (i = 0; i < 256; ++i) {

        kvm_irqchip_add_irq_route(kvm_state, i, 0, i);

    }



    kvm_irqfds_allowed = true;

    kvm_msi_via_irqfd_allowed = true;

    kvm_gsi_routing_allowed = true;



    kvm_irqchip_commit_routes(s);

}
