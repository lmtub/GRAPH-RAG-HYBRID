static PCIBus *pci_get_bus_devfn(int *devfnp, const char *devaddr)

{

    int dom, bus;

    unsigned slot;



    if (!devaddr) {

        *devfnp = -1;

        return pci_find_bus(0);

    }



    if (pci_parse_devaddr(devaddr, &dom, &bus, &slot) < 0) {

        return NULL;

    }



    *devfnp = slot << 3;

    return pci_find_bus(bus);

}
