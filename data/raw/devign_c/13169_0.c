static int pci_bridge_initfn(PCIDevice *dev)

{

    PCIBridge *s = DO_UPCAST(PCIBridge, dev, dev);



    pci_config_set_vendor_id(s->dev.config, s->vid);

    pci_config_set_device_id(s->dev.config, s->did);



    s->dev.config[0x04] = 0x06; // command = bus master, pci mem

    s->dev.config[0x05] = 0x00;

    s->dev.config[0x06] = 0xa0; // status = fast back-to-back, 66MHz, no error

    s->dev.config[0x07] = 0x00; // status = fast devsel

    s->dev.config[0x08] = 0x00; // revision

    s->dev.config[0x09] = 0x00; // programming i/f

    pci_config_set_class(s->dev.config, PCI_CLASS_BRIDGE_PCI);

    s->dev.config[0x0D] = 0x10; // latency_timer

    s->dev.config[PCI_HEADER_TYPE] =

        PCI_HEADER_TYPE_MULTI_FUNCTION | PCI_HEADER_TYPE_BRIDGE; // header_type

    s->dev.config[0x1E] = 0xa0; // secondary status

    return 0;

}
