static void unplug_nic(PCIBus *b, PCIDevice *d)

{

    if (pci_get_word(d->config + PCI_CLASS_DEVICE) ==

            PCI_CLASS_NETWORK_ETHERNET) {

        qdev_unplug(&(d->qdev), NULL);

    }

}
