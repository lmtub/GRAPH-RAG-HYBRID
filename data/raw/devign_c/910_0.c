static int usb_uhci_piix4_initfn(PCIDevice *dev)

{

    UHCIState *s = DO_UPCAST(UHCIState, dev, dev);

    uint8_t *pci_conf = s->dev.config;



    pci_config_set_vendor_id(pci_conf, PCI_VENDOR_ID_INTEL);

    pci_config_set_device_id(pci_conf, PCI_DEVICE_ID_INTEL_82371AB_2);

    return usb_uhci_common_initfn(s);

}
