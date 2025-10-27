uint32_t msix_bar_size(PCIDevice *dev)

{

    return (dev->cap_present & QEMU_PCI_CAP_MSIX) ?

        dev->msix_bar_size : 0;

}
