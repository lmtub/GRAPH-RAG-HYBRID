QVirtioPCIDevice *qvirtio_pci_device_find(QPCIBus *bus, uint16_t device_type)

{

    QVirtioPCIDevice *dev = NULL;

    qvirtio_pci_foreach(bus, device_type, qvirtio_pci_assign_device, &dev);



    dev->vdev.bus = &qvirtio_pci;



    return dev;

}
