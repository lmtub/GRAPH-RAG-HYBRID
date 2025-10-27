static void pci_qdev_realize(DeviceState *qdev, Error **errp)

{

    PCIDevice *pci_dev = (PCIDevice *)qdev;

    PCIDeviceClass *pc = PCI_DEVICE_GET_CLASS(pci_dev);

    Error *local_err = NULL;

    PCIBus *bus;

    bool is_default_rom;



    /* initialize cap_present for pci_is_express() and pci_config_size() */

    if (pc->is_express) {

        pci_dev->cap_present |= QEMU_PCI_CAP_EXPRESS;

    }



    bus = PCI_BUS(qdev_get_parent_bus(qdev));

    pci_dev = do_pci_register_device(pci_dev, bus,

                                     object_get_typename(OBJECT(qdev)),

                                     pci_dev->devfn, errp);

    if (pci_dev == NULL)

        return;



    if (pc->realize) {

        pc->realize(pci_dev, &local_err);

        if (local_err) {

            error_propagate(errp, local_err);

            do_pci_unregister_device(pci_dev);

            return;

        }

    }



    /* rom loading */

    is_default_rom = false;

    if (pci_dev->romfile == NULL && pc->romfile != NULL) {

        pci_dev->romfile = g_strdup(pc->romfile);

        is_default_rom = true;

    }



    pci_add_option_rom(pci_dev, is_default_rom, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        pci_qdev_unrealize(DEVICE(pci_dev), NULL);

        return;

    }

}
