static void virtio_ioport_write(void *opaque, uint32_t addr, uint32_t val)

{

    VirtIOPCIProxy *proxy = opaque;

    VirtIODevice *vdev = proxy->vdev;

    target_phys_addr_t pa;



    switch (addr) {

    case VIRTIO_PCI_GUEST_FEATURES:

	/* Guest does not negotiate properly?  We have to assume nothing. */

	if (val & (1 << VIRTIO_F_BAD_FEATURE)) {

	    if (vdev->bad_features)

		val = proxy->host_features & vdev->bad_features(vdev);

	    else

		val = 0;

	}

        if (vdev->set_features)

            vdev->set_features(vdev, val);

        vdev->guest_features = val;

        break;

    case VIRTIO_PCI_QUEUE_PFN:

        pa = (target_phys_addr_t)val << VIRTIO_PCI_QUEUE_ADDR_SHIFT;

        if (pa == 0) {

            virtio_pci_stop_ioeventfd(proxy);

            virtio_reset(proxy->vdev);

            msix_unuse_all_vectors(&proxy->pci_dev);

        }

        else

            virtio_queue_set_addr(vdev, vdev->queue_sel, pa);

        break;

    case VIRTIO_PCI_QUEUE_SEL:

        if (val < VIRTIO_PCI_QUEUE_MAX)

            vdev->queue_sel = val;

        break;

    case VIRTIO_PCI_QUEUE_NOTIFY:

        virtio_queue_notify(vdev, val);

        break;

    case VIRTIO_PCI_STATUS:

        if (!(val & VIRTIO_CONFIG_S_DRIVER_OK)) {

            virtio_pci_stop_ioeventfd(proxy);

        }



        virtio_set_status(vdev, val & 0xFF);



        if (val & VIRTIO_CONFIG_S_DRIVER_OK) {

            virtio_pci_start_ioeventfd(proxy);

        }



        if (vdev->status == 0) {

            virtio_reset(proxy->vdev);

            msix_unuse_all_vectors(&proxy->pci_dev);

        }



        /* Linux before 2.6.34 sets the device as OK without enabling

           the PCI device bus master bit. In this case we need to disable

           some safety checks. */

        if ((val & VIRTIO_CONFIG_S_DRIVER_OK) &&

            !(proxy->pci_dev.config[PCI_COMMAND] & PCI_COMMAND_MASTER)) {

            proxy->flags |= VIRTIO_PCI_FLAG_BUS_MASTER_BUG;

        }

        break;

    case VIRTIO_MSI_CONFIG_VECTOR:

        msix_vector_unuse(&proxy->pci_dev, vdev->config_vector);

        /* Make it possible for guest to discover an error took place. */

        if (msix_vector_use(&proxy->pci_dev, val) < 0)

            val = VIRTIO_NO_VECTOR;

        vdev->config_vector = val;

        break;

    case VIRTIO_MSI_QUEUE_VECTOR:

        msix_vector_unuse(&proxy->pci_dev,

                          virtio_queue_vector(vdev, vdev->queue_sel));

        /* Make it possible for guest to discover an error took place. */

        if (msix_vector_use(&proxy->pci_dev, val) < 0)

            val = VIRTIO_NO_VECTOR;

        virtio_queue_set_vector(vdev, vdev->queue_sel, val);

        break;

    default:

        error_report("%s: unexpected address 0x%x value 0x%x",

                     __func__, addr, val);

        break;

    }

}
