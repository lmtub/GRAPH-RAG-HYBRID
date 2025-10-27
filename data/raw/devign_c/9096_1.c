void pci_cmd646_ide_init(PCIBus *bus, DriveInfo **hd_table,

                         int secondary_ide_enabled)

{

    PCIDevice *dev;



    dev = pci_create(bus, -1, "CMD646 IDE");

    qdev_prop_set_uint32(&dev->qdev, "secondary", secondary_ide_enabled);

    qdev_init(&dev->qdev);



    pci_ide_create_devs(dev, hd_table);

}
