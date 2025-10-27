static int trap_msix(S390PCIBusDevice *pbdev, uint64_t offset, uint8_t pcias)

{

    if (pbdev->msix.available && pbdev->msix.table_bar == pcias &&

        offset >= pbdev->msix.table_offset &&

        offset <= pbdev->msix.table_offset +

                  (pbdev->msix.entries - 1) * PCI_MSIX_ENTRY_SIZE) {

        return 1;

    } else {

        return 0;

    }

}
