static void vfio_disable_msix(VFIODevice *vdev)
{
    msix_unset_vector_notifiers(&vdev->pdev);
    if (vdev->nr_vectors) {
        vfio_disable_irqindex(vdev, VFIO_PCI_MSIX_IRQ_INDEX);
    vfio_disable_msi_common(vdev);
    DPRINTF("%s(%04x:%02x:%02x.%x)\n", __func__, vdev->host.domain,
            vdev->host.bus, vdev->host.slot, vdev->host.function);