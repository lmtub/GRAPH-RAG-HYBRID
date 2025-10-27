static void vfio_intx_enable_kvm(VFIOPCIDevice *vdev)

{

#ifdef CONFIG_KVM

    struct kvm_irqfd irqfd = {

        .fd = event_notifier_get_fd(&vdev->intx.interrupt),

        .gsi = vdev->intx.route.irq,

        .flags = KVM_IRQFD_FLAG_RESAMPLE,

    };

    struct vfio_irq_set *irq_set;

    int ret, argsz;

    int32_t *pfd;



    if (!VFIO_ALLOW_KVM_INTX || !kvm_irqfds_enabled() ||

        vdev->intx.route.mode != PCI_INTX_ENABLED ||

        !kvm_resamplefds_enabled()) {

        return;

    }



    /* Get to a known interrupt state */

    qemu_set_fd_handler(irqfd.fd, NULL, NULL, vdev);

    vfio_mask_single_irqindex(&vdev->vbasedev, VFIO_PCI_INTX_IRQ_INDEX);

    vdev->intx.pending = false;

    pci_irq_deassert(&vdev->pdev);



    /* Get an eventfd for resample/unmask */

    if (event_notifier_init(&vdev->intx.unmask, 0)) {

        error_report("vfio: Error: event_notifier_init failed eoi");

        goto fail;

    }



    /* KVM triggers it, VFIO listens for it */

    irqfd.resamplefd = event_notifier_get_fd(&vdev->intx.unmask);



    if (kvm_vm_ioctl(kvm_state, KVM_IRQFD, &irqfd)) {

        error_report("vfio: Error: Failed to setup resample irqfd: %m");

        goto fail_irqfd;

    }



    argsz = sizeof(*irq_set) + sizeof(*pfd);



    irq_set = g_malloc0(argsz);

    irq_set->argsz = argsz;

    irq_set->flags = VFIO_IRQ_SET_DATA_EVENTFD | VFIO_IRQ_SET_ACTION_UNMASK;

    irq_set->index = VFIO_PCI_INTX_IRQ_INDEX;

    irq_set->start = 0;

    irq_set->count = 1;

    pfd = (int32_t *)&irq_set->data;



    *pfd = irqfd.resamplefd;



    ret = ioctl(vdev->vbasedev.fd, VFIO_DEVICE_SET_IRQS, irq_set);

    g_free(irq_set);

    if (ret) {

        error_report("vfio: Error: Failed to setup INTx unmask fd: %m");

        goto fail_vfio;

    }



    /* Let'em rip */

    vfio_unmask_single_irqindex(&vdev->vbasedev, VFIO_PCI_INTX_IRQ_INDEX);



    vdev->intx.kvm_accel = true;



    trace_vfio_intx_enable_kvm(vdev->vbasedev.name);



    return;



fail_vfio:

    irqfd.flags = KVM_IRQFD_FLAG_DEASSIGN;

    kvm_vm_ioctl(kvm_state, KVM_IRQFD, &irqfd);

fail_irqfd:

    event_notifier_cleanup(&vdev->intx.unmask);

fail:

    qemu_set_fd_handler(irqfd.fd, vfio_intx_interrupt, NULL, vdev);

    vfio_unmask_single_irqindex(&vdev->vbasedev, VFIO_PCI_INTX_IRQ_INDEX);

#endif

}
