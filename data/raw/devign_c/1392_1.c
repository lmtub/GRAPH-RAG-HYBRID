static void vfio_disable_interrupts(VFIOPCIDevice *vdev)

{

    switch (vdev->interrupt) {

    case VFIO_INT_INTx:

        vfio_disable_intx(vdev);

        break;

    case VFIO_INT_MSI:

        vfio_disable_msi(vdev);

        break;

    case VFIO_INT_MSIX:

        vfio_disable_msix(vdev);

        break;

    }

}
