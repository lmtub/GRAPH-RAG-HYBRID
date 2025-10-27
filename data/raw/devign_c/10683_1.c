void qvirtio_pci_foreach(QPCIBus *bus, uint16_t device_type,

                void (*func)(QVirtioDevice *d, void *data), void *data)

{

    QVirtioPCIForeachData d = { .func = func,

                                .device_type = device_type,

                                .user_data = data };



    qpci_device_foreach(bus, PCI_VENDOR_ID_REDHAT_QUMRANET, -1,

                                qvirtio_pci_foreach_callback, &d);

}
