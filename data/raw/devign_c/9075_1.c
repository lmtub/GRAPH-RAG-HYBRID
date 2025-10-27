static int assign_intx(AssignedDevice *dev, Error **errp)

{

    AssignedIRQType new_type;

    PCIINTxRoute intx_route;

    bool intx_host_msi;

    int r;

    Error *local_err = NULL;



    /* Interrupt PIN 0 means don't use INTx */

    if (assigned_dev_pci_read_byte(&dev->dev, PCI_INTERRUPT_PIN) == 0) {

        pci_device_set_intx_routing_notifier(&dev->dev, NULL);

        return 0;

    }



    verify_irqchip_in_kernel(&local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        return -ENOTSUP;

    }



    pci_device_set_intx_routing_notifier(&dev->dev,

                                         assigned_dev_update_irq_routing);



    intx_route = pci_device_route_intx_to_irq(&dev->dev, dev->intpin);

    assert(intx_route.mode != PCI_INTX_INVERTED);



    if (!pci_intx_route_changed(&dev->intx_route, &intx_route)) {

        return 0;

    }



    switch (dev->assigned_irq_type) {

    case ASSIGNED_IRQ_INTX_HOST_INTX:

    case ASSIGNED_IRQ_INTX_HOST_MSI:

        intx_host_msi = dev->assigned_irq_type == ASSIGNED_IRQ_INTX_HOST_MSI;

        r = kvm_device_intx_deassign(kvm_state, dev->dev_id, intx_host_msi);

        break;

    case ASSIGNED_IRQ_MSI:

        r = kvm_device_msi_deassign(kvm_state, dev->dev_id);

        break;

    case ASSIGNED_IRQ_MSIX:

        r = kvm_device_msix_deassign(kvm_state, dev->dev_id);

        break;

    default:

        r = 0;

        break;

    }

    if (r) {

        perror("assign_intx: deassignment of previous interrupt failed");

    }

    dev->assigned_irq_type = ASSIGNED_IRQ_NONE;



    if (intx_route.mode == PCI_INTX_DISABLED) {

        dev->intx_route = intx_route;

        return 0;

    }



retry:

    if (dev->features & ASSIGNED_DEVICE_PREFER_MSI_MASK &&

        dev->cap.available & ASSIGNED_DEVICE_CAP_MSI) {

        intx_host_msi = true;

        new_type = ASSIGNED_IRQ_INTX_HOST_MSI;

    } else {

        intx_host_msi = false;

        new_type = ASSIGNED_IRQ_INTX_HOST_INTX;

    }



    r = kvm_device_intx_assign(kvm_state, dev->dev_id, intx_host_msi,

                               intx_route.irq);

    if (r < 0) {

        if (r == -EIO && !(dev->features & ASSIGNED_DEVICE_PREFER_MSI_MASK) &&

            dev->cap.available & ASSIGNED_DEVICE_CAP_MSI) {

            /* Retry with host-side MSI. There might be an IRQ conflict and

             * either the kernel or the device doesn't support sharing. */

            error_report("Host-side INTx sharing not supported, "

                         "using MSI instead");

            error_printf("Some devices do not work properly in this mode.\n");

            dev->features |= ASSIGNED_DEVICE_PREFER_MSI_MASK;

            goto retry;

        }

        error_setg_errno(errp, -r, "Failed to assign irq for \"%s\"",

                         dev->dev.qdev.id);

        error_append_hint(errp, "Perhaps you are assigning a device "

                          "that shares an IRQ with another device?\n");

        return r;

    }



    dev->intx_route = intx_route;

    dev->assigned_irq_type = new_type;

    return r;

}
