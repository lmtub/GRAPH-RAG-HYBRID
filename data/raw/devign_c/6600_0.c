static int vfio_msix_vector_do_use(PCIDevice *pdev, unsigned int nr,

                                   MSIMessage *msg, IOHandler *handler)

{

    VFIOPCIDevice *vdev = DO_UPCAST(VFIOPCIDevice, pdev, pdev);

    VFIOMSIVector *vector;

    int ret;



    trace_vfio_msix_vector_do_use(vdev->vbasedev.name, nr);



    vector = &vdev->msi_vectors[nr];



    if (!vector->use) {

        vector->vdev = vdev;

        vector->virq = -1;

        if (event_notifier_init(&vector->interrupt, 0)) {

            error_report("vfio: Error: event_notifier_init failed");

        }

        vector->use = true;

        msix_vector_use(pdev, nr);

    }



    qemu_set_fd_handler(event_notifier_get_fd(&vector->interrupt),

                        handler, NULL, vector);



    /*

     * Attempt to enable route through KVM irqchip,

     * default to userspace handling if unavailable.

     */

    if (vector->virq >= 0) {

        if (!msg) {

            vfio_remove_kvm_msi_virq(vector);

        } else {

            vfio_update_kvm_msi_virq(vector, *msg);

        }

    } else {

        vfio_add_kvm_msi_virq(vector, msg, true);

    }



    /*

     * We don't want to have the host allocate all possible MSI vectors

     * for a device if they're not in use, so we shutdown and incrementally

     * increase them as needed.

     */

    if (vdev->nr_vectors < nr + 1) {

        vfio_disable_irqindex(&vdev->vbasedev, VFIO_PCI_MSIX_IRQ_INDEX);

        vdev->nr_vectors = nr + 1;

        ret = vfio_enable_vectors(vdev, true);

        if (ret) {

            error_report("vfio: failed to enable vectors, %d", ret);

        }

    } else {

        int argsz;

        struct vfio_irq_set *irq_set;

        int32_t *pfd;



        argsz = sizeof(*irq_set) + sizeof(*pfd);



        irq_set = g_malloc0(argsz);

        irq_set->argsz = argsz;

        irq_set->flags = VFIO_IRQ_SET_DATA_EVENTFD |

                         VFIO_IRQ_SET_ACTION_TRIGGER;

        irq_set->index = VFIO_PCI_MSIX_IRQ_INDEX;

        irq_set->start = nr;

        irq_set->count = 1;

        pfd = (int32_t *)&irq_set->data;



        if (vector->virq >= 0) {

            *pfd = event_notifier_get_fd(&vector->kvm_interrupt);

        } else {

            *pfd = event_notifier_get_fd(&vector->interrupt);

        }



        ret = ioctl(vdev->vbasedev.fd, VFIO_DEVICE_SET_IRQS, irq_set);

        g_free(irq_set);

        if (ret) {

            error_report("vfio: failed to modify vector, %d", ret);

        }

    }



    return 0;

}
