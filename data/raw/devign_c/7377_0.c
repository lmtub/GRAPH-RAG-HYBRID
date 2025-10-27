static int pci_piix3_ide_initfn(PCIDevice *dev)

{

    PCIIDEState *d = DO_UPCAST(PCIIDEState, dev, dev);



    pci_config_set_vendor_id(d->dev.config, PCI_VENDOR_ID_INTEL);

    pci_config_set_device_id(d->dev.config, PCI_DEVICE_ID_INTEL_82371SB_1);

    return pci_piix_ide_initfn(d);

}
