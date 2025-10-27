void msi_uninit(struct PCIDevice *dev)

{

    uint16_t flags;

    uint8_t cap_size;



    if (!(dev->cap_present & QEMU_PCI_CAP_MSI)) {

        return;

    }

    flags = pci_get_word(dev->config + msi_flags_off(dev));

    cap_size = msi_cap_sizeof(flags);

    pci_del_capability(dev, PCI_CAP_ID_MSI, cap_size);

    dev->cap_present &= ~QEMU_PCI_CAP_MSI;



    MSI_DEV_PRINTF(dev, "uninit\n");

}
