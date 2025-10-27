static int piix4_initfn(PCIDevice *dev)

{

    PIIX4State *d = DO_UPCAST(PIIX4State, dev, dev);

    uint8_t *pci_conf;



    isa_bus_new(&d->dev.qdev);



    pci_conf = d->dev.config;

    pci_config_set_vendor_id(pci_conf, PCI_VENDOR_ID_INTEL);

    pci_config_set_device_id(pci_conf, PCI_DEVICE_ID_INTEL_82371AB_0); // 82371AB/EB/MB PIIX4 PCI-to-ISA bridge

    pci_config_set_class(pci_conf, PCI_CLASS_BRIDGE_ISA);



    piix4_dev = &d->dev;

    qemu_register_reset(piix4_reset, d);

    return 0;

}
