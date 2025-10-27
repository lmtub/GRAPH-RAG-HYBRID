static int pci_qdev_init(DeviceState *qdev)

{

    PCIDevice *pci_dev = (PCIDevice *)qdev;

    PCIDeviceClass *pc = PCI_DEVICE_GET_CLASS(pci_dev);

    PCIBus *bus;

    int rc;

    bool is_default_rom;



    /* initialize cap_present for pci_is_express() and pci_config_size() */

    if (pc->is_express) {

        pci_dev->cap_present |= QEMU_PCI_CAP_EXPRESS;

    }



    bus = PCI_BUS(qdev_get_parent_bus(qdev));

    pci_dev = do_pci_register_device(pci_dev, bus,

                                     object_get_typename(OBJECT(qdev)),

                                     pci_dev->devfn);

    if (pci_dev == NULL)

        return -1;



    if (pc->init) {

        rc = pc->init(pci_dev);

        if (rc != 0) {

            do_pci_unregister_device(pci_dev);

            return rc;

        }

    }



    /* rom loading */

    is_default_rom = false;

    if (pci_dev->romfile == NULL && pc->romfile != NULL) {

        pci_dev->romfile = g_strdup(pc->romfile);

        is_default_rom = true;

    }

    pci_add_option_rom(pci_dev, is_default_rom);



    return 0;

}
