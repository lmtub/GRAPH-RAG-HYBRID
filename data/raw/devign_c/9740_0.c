static uint32_t vfio_pci_read_config(PCIDevice *pdev, uint32_t addr, int len)

{

    VFIODevice *vdev = DO_UPCAST(VFIODevice, pdev, pdev);

    uint32_t val = 0;



    /*

     * We only need QEMU PCI config support for the ROM BAR, the MSI and MSIX

     * capabilities, and the multifunction bit below.  We let VFIO handle

     * virtualizing everything else.  Performance is not a concern here.

     */

    if (ranges_overlap(addr, len, PCI_ROM_ADDRESS, 4) ||

        (pdev->cap_present & QEMU_PCI_CAP_MSIX &&

         ranges_overlap(addr, len, pdev->msix_cap, MSIX_CAP_LENGTH)) ||

        (pdev->cap_present & QEMU_PCI_CAP_MSI &&

         ranges_overlap(addr, len, pdev->msi_cap, vdev->msi_cap_size))) {



        val = pci_default_read_config(pdev, addr, len);

    } else {

        if (pread(vdev->fd, &val, len, vdev->config_offset + addr) != len) {

            error_report("%s(%04x:%02x:%02x.%x, 0x%x, 0x%x) failed: %m",

                         __func__, vdev->host.domain, vdev->host.bus,

                         vdev->host.slot, vdev->host.function, addr, len);

            return -errno;

        }

        val = le32_to_cpu(val);

    }



    /* Multifunction bit is virualized in QEMU */

    if (unlikely(ranges_overlap(addr, len, PCI_HEADER_TYPE, 1))) {

        uint32_t mask = PCI_HEADER_TYPE_MULTI_FUNCTION;



        if (len == 4) {

            mask <<= 16;

        }



        if (pdev->cap_present & QEMU_PCI_CAP_MULTIFUNCTION) {

            val |= mask;

        } else {

            val &= ~mask;

        }

    }



    DPRINTF("%s(%04x:%02x:%02x.%x, @0x%x, len=0x%x) %x\n", __func__,

            vdev->host.domain, vdev->host.bus, vdev->host.slot,

            vdev->host.function, addr, len, val);



    return val;

}
