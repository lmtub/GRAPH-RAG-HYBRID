PCIDevice *pci_nic_init(NICInfo *nd, const char *default_model,

                        const char *default_devaddr)

{

    const char *devaddr = nd->devaddr ? nd->devaddr : default_devaddr;

    PCIDevice *pci_dev;

    DeviceState *dev;

    int i;



    i = qemu_check_nic_model_list(nd, pci_nic_models, default_model);

    pci_dev = pci_create(pci_nic_names[i], devaddr);

    dev = &pci_dev->qdev;

    if (nd->id)

        dev->id = qemu_strdup(nd->id);

    dev->nd = nd;

    qdev_init(dev);

    nd->private = dev;

    return pci_dev;

}
