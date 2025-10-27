static int vfio_early_setup_msix(VFIOPCIDevice *vdev)

{

    uint8_t pos;

    uint16_t ctrl;

    uint32_t table, pba;

    int fd = vdev->vbasedev.fd;



    pos = pci_find_capability(&vdev->pdev, PCI_CAP_ID_MSIX);

    if (!pos) {

        return 0;

    }



    if (pread(fd, &ctrl, sizeof(ctrl),

              vdev->config_offset + pos + PCI_CAP_FLAGS) != sizeof(ctrl)) {

        return -errno;

    }



    if (pread(fd, &table, sizeof(table),

              vdev->config_offset + pos + PCI_MSIX_TABLE) != sizeof(table)) {

        return -errno;

    }



    if (pread(fd, &pba, sizeof(pba),

              vdev->config_offset + pos + PCI_MSIX_PBA) != sizeof(pba)) {

        return -errno;

    }



    ctrl = le16_to_cpu(ctrl);

    table = le32_to_cpu(table);

    pba = le32_to_cpu(pba);



    vdev->msix = g_malloc0(sizeof(*(vdev->msix)));

    vdev->msix->table_bar = table & PCI_MSIX_FLAGS_BIRMASK;

    vdev->msix->table_offset = table & ~PCI_MSIX_FLAGS_BIRMASK;

    vdev->msix->pba_bar = pba & PCI_MSIX_FLAGS_BIRMASK;

    vdev->msix->pba_offset = pba & ~PCI_MSIX_FLAGS_BIRMASK;

    vdev->msix->entries = (ctrl & PCI_MSIX_FLAGS_QSIZE) + 1;



    /*

     * Test the size of the pba_offset variable and catch if it extends outside

     * of the specified BAR. If it is the case, we need to apply a hardware

     * specific quirk if the device is known or we have a broken configuration.

     */

    if (vdev->msix->pba_offset >=

        vdev->bars[vdev->msix->pba_bar].region.size) {



        PCIDevice *pdev = &vdev->pdev;

        uint16_t vendor = pci_get_word(pdev->config + PCI_VENDOR_ID);

        uint16_t device = pci_get_word(pdev->config + PCI_DEVICE_ID);



        /*

         * Chelsio T5 Virtual Function devices are encoded as 0x58xx for T5

         * adapters. The T5 hardware returns an incorrect value of 0x8000 for

         * the VF PBA offset while the BAR itself is only 8k. The correct value

         * is 0x1000, so we hard code that here.

         */

        if (vendor == PCI_VENDOR_ID_CHELSIO && (device & 0xff00) == 0x5800) {

            vdev->msix->pba_offset = 0x1000;

        } else {

            error_report("vfio: Hardware reports invalid configuration, "

                         "MSIX PBA outside of specified BAR");

            return -EINVAL;

        }

    }



    trace_vfio_early_setup_msix(vdev->vbasedev.name, pos,

                                vdev->msix->table_bar,

                                vdev->msix->table_offset,

                                vdev->msix->entries);



    return 0;

}
