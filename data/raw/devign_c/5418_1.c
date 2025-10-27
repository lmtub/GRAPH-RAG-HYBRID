static int vfio_msix_setup(VFIOPCIDevice *vdev, int pos, Error **errp)

{

    int ret;



    vdev->msix->pending = g_malloc0(BITS_TO_LONGS(vdev->msix->entries) *

                                    sizeof(unsigned long));

    ret = msix_init(&vdev->pdev, vdev->msix->entries,

                    vdev->bars[vdev->msix->table_bar].region.mem,

                    vdev->msix->table_bar, vdev->msix->table_offset,

                    vdev->bars[vdev->msix->pba_bar].region.mem,

                    vdev->msix->pba_bar, vdev->msix->pba_offset, pos);

    if (ret < 0) {

        if (ret == -ENOTSUP) {

            return 0;

        }

        error_setg(errp, "msix_init failed");

        return ret;

    }



    /*

     * The PCI spec suggests that devices provide additional alignment for

     * MSI-X structures and avoid overlapping non-MSI-X related registers.

     * For an assigned device, this hopefully means that emulation of MSI-X

     * structures does not affect the performance of the device.  If devices

     * fail to provide that alignment, a significant performance penalty may

     * result, for instance Mellanox MT27500 VFs:

     * http://www.spinics.net/lists/kvm/msg125881.html

     *

     * The PBA is simply not that important for such a serious regression and

     * most drivers do not appear to look at it.  The solution for this is to

     * disable the PBA MemoryRegion unless it's being used.  We disable it

     * here and only enable it if a masked vector fires through QEMU.  As the

     * vector-use notifier is called, which occurs on unmask, we test whether

     * PBA emulation is needed and again disable if not.

     */

    memory_region_set_enabled(&vdev->pdev.msix_pba_mmio, false);



    return 0;

}
