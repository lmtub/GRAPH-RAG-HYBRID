static bool vfio_pci_host_match(PCIHostDeviceAddress *host1,

                                PCIHostDeviceAddress *host2)

{

    return (host1->domain == host2->domain && host1->bus == host2->bus &&

            host1->slot == host2->slot && host1->function == host2->function);

}
