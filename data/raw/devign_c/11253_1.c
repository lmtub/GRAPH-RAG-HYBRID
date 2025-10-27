int pci_vga_init(PCIBus *bus,

                 unsigned long vga_bios_offset, int vga_bios_size)

{

    PCIDevice *dev;



    dev = pci_create(bus, -1, "VGA");

    qdev_prop_set_uint32(&dev->qdev, "bios-offset", vga_bios_offset);

    qdev_prop_set_uint32(&dev->qdev, "bios-size", vga_bios_offset);

    qdev_init(&dev->qdev);



    return 0;

}
