static QPCIDevice *get_pci_device(void **bmdma_base, void **ide_base)

{

    QPCIDevice *dev;

    uint16_t vendor_id, device_id;



    if (!pcibus) {

        pcibus = qpci_init_pc(NULL);

    }



    /* Find PCI device and verify it's the right one */

    dev = qpci_device_find(pcibus, QPCI_DEVFN(IDE_PCI_DEV, IDE_PCI_FUNC));

    g_assert(dev != NULL);



    vendor_id = qpci_config_readw(dev, PCI_VENDOR_ID);

    device_id = qpci_config_readw(dev, PCI_DEVICE_ID);

    g_assert(vendor_id == PCI_VENDOR_ID_INTEL);

    g_assert(device_id == PCI_DEVICE_ID_INTEL_82371SB_1);



    /* Map bmdma BAR */

    *bmdma_base = qpci_iomap(dev, 4, NULL);



    *ide_base = qpci_legacy_iomap(dev, IDE_BASE);



    qpci_device_enable(dev);



    return dev;

}
