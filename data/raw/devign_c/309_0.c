PCIDevice *pci_register_device(PCIBus *bus, const char *name,

                               int instance_size, int devfn,

                               PCIConfigReadFunc *config_read,

                               PCIConfigWriteFunc *config_write)

{

    PCIDevice *pci_dev;



    pci_dev = qemu_mallocz(instance_size);

    pci_dev = do_pci_register_device(pci_dev, bus, name, devfn,

                                     config_read, config_write);

    return pci_dev;

}
