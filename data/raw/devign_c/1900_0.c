static int vfio_enable_intx(VFIODevice *vdev)

{

    VFIOIRQSetFD irq_set_fd = {

        .irq_set = {

            .argsz = sizeof(irq_set_fd),

            .flags = VFIO_IRQ_SET_DATA_EVENTFD | VFIO_IRQ_SET_ACTION_TRIGGER,

            .index = VFIO_PCI_INTX_IRQ_INDEX,

            .start = 0,

            .count = 1,

        },

    };

    uint8_t pin = vfio_pci_read_config(&vdev->pdev, PCI_INTERRUPT_PIN, 1);

    int ret;



    if (vdev->intx.disabled || !pin) {

        return 0;

    }



    vfio_disable_interrupts(vdev);



    vdev->intx.pin = pin - 1; /* Pin A (1) -> irq[0] */

    ret = event_notifier_init(&vdev->intx.interrupt, 0);

    if (ret) {

        error_report("vfio: Error: event_notifier_init failed\n");

        return ret;

    }



    irq_set_fd.fd = event_notifier_get_fd(&vdev->intx.interrupt);

    qemu_set_fd_handler(irq_set_fd.fd, vfio_intx_interrupt, NULL, vdev);



    if (ioctl(vdev->fd, VFIO_DEVICE_SET_IRQS, &irq_set_fd)) {

        error_report("vfio: Error: Failed to setup INTx fd: %m\n");

        return -errno;

    }



    /*

     * Disable mmaps so we can trap on BAR accesses.  We interpret any

     * access as a response to an interrupt and unmask the physical

     * device.  The device will re-assert if the interrupt is still

     * pending.  We'll likely retrigger on the host multiple times per

     * guest interrupt, but without EOI notification it's better than

     * nothing.  Acceleration paths through KVM will avoid this.

     */

    vfio_mmap_set_enabled(vdev, false);



    vdev->interrupt = VFIO_INT_INTx;



    DPRINTF("%s(%04x:%02x:%02x.%x)\n", __func__, vdev->host.domain,

            vdev->host.bus, vdev->host.slot, vdev->host.function);



    return 0;

}
