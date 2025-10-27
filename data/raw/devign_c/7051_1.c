PCIBus *pci_get_bus_devfn(int *devfnp, PCIBus *root, const char *devaddr)

{

    int dom, bus;

    unsigned slot;



    assert(!root->parent_dev);



    if (!root) {

        fprintf(stderr, "No primary PCI bus\n");

        return NULL;

    }



    if (!devaddr) {

        *devfnp = -1;

        return pci_find_bus_nr(root, 0);

    }



    if (pci_parse_devaddr(devaddr, &dom, &bus, &slot, NULL) < 0) {

        return NULL;

    }



    if (dom != 0) {

        fprintf(stderr, "No support for non-zero PCI domains\n");

        return NULL;

    }



    *devfnp = PCI_DEVFN(slot, 0);

    return pci_find_bus_nr(root, bus);

}
