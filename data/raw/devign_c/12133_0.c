static void s390_pcihost_hot_plug(HotplugHandler *hotplug_dev,

                                  DeviceState *dev, Error **errp)

{

    PCIDevice *pci_dev = PCI_DEVICE(dev);

    S390PCIBusDevice *pbdev;

    S390pciState *s = S390_PCI_HOST_BRIDGE(pci_device_root_bus(pci_dev)

                                           ->qbus.parent);



    pbdev = &s->pbdev[PCI_SLOT(pci_dev->devfn)];



    pbdev->fid = s390_pci_get_pfid(pci_dev);

    pbdev->pdev = pci_dev;

    pbdev->configured = true;

    pbdev->fh = s390_pci_get_pfh(pci_dev);



    s390_pcihost_setup_msix(pbdev);



    if (dev->hotplugged) {

        s390_pci_generate_plug_event(HP_EVENT_RESERVED_TO_STANDBY,

                                     pbdev->fh, pbdev->fid);

        s390_pci_generate_plug_event(HP_EVENT_TO_CONFIGURED,

                                     pbdev->fh, pbdev->fid);

    }

}
