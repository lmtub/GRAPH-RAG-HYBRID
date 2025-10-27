static void vfio_msi_enable(VFIOPCIDevice *vdev)

{

    int ret, i;



    vfio_disable_interrupts(vdev);



    vdev->nr_vectors = msi_nr_vectors_allocated(&vdev->pdev);

retry:

    vdev->msi_vectors = g_malloc0(vdev->nr_vectors * sizeof(VFIOMSIVector));



    for (i = 0; i < vdev->nr_vectors; i++) {

        VFIOMSIVector *vector = &vdev->msi_vectors[i];

        MSIMessage msg = msi_get_message(&vdev->pdev, i);



        vector->vdev = vdev;

        vector->virq = -1;

        vector->use = true;



        if (event_notifier_init(&vector->interrupt, 0)) {

            error_report("vfio: Error: event_notifier_init failed");

        }



        qemu_set_fd_handler(event_notifier_get_fd(&vector->interrupt),

                            vfio_msi_interrupt, NULL, vector);



        /*

         * Attempt to enable route through KVM irqchip,

         * default to userspace handling if unavailable.

         */

        vfio_add_kvm_msi_virq(vdev, vector, &msg, false);

    }



    /* Set interrupt type prior to possible interrupts */

    vdev->interrupt = VFIO_INT_MSI;



    ret = vfio_enable_vectors(vdev, false);

    if (ret) {

        if (ret < 0) {

            error_report("vfio: Error: Failed to setup MSI fds: %m");

        } else if (ret != vdev->nr_vectors) {

            error_report("vfio: Error: Failed to enable %d "

                         "MSI vectors, retry with %d", vdev->nr_vectors, ret);

        }



        for (i = 0; i < vdev->nr_vectors; i++) {

            VFIOMSIVector *vector = &vdev->msi_vectors[i];

            if (vector->virq >= 0) {

                vfio_remove_kvm_msi_virq(vector);

            }

            qemu_set_fd_handler(event_notifier_get_fd(&vector->interrupt),

                                NULL, NULL, NULL);

            event_notifier_cleanup(&vector->interrupt);

        }



        g_free(vdev->msi_vectors);



        if (ret > 0 && ret != vdev->nr_vectors) {

            vdev->nr_vectors = ret;

            goto retry;

        }

        vdev->nr_vectors = 0;



        /*

         * Failing to setup MSI doesn't really fall within any specification.

         * Let's try leaving interrupts disabled and hope the guest figures

         * out to fall back to INTx for this device.

         */

        error_report("vfio: Error: Failed to enable MSI");

        vdev->interrupt = VFIO_INT_NONE;



        return;

    }



    trace_vfio_msi_enable(vdev->vbasedev.name, vdev->nr_vectors);

}
