static void pci_nic_uninit(PCIDevice *pci_dev)

{

    EEPRO100State *s = DO_UPCAST(EEPRO100State, dev, pci_dev);



    vmstate_unregister(&pci_dev->qdev, s->vmstate, s);


    eeprom93xx_free(&pci_dev->qdev, s->eeprom);

    qemu_del_nic(s->nic);

}