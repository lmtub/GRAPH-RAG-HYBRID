static void pci_qdev_unrealize(DeviceState *dev, Error **errp)

{

    PCIDevice *pci_dev = PCI_DEVICE(dev);

    PCIDeviceClass *pc = PCI_DEVICE_GET_CLASS(pci_dev);



    pci_unregister_io_regions(pci_dev);

    pci_del_option_rom(pci_dev);



    if (pc->exit) {

        pc->exit(pci_dev);

    }




    do_pci_unregister_device(pci_dev);

}