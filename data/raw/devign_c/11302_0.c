static int spapr_create_pci_child_dt(sPAPRPHBState *phb, PCIDevice *dev,

                                     int drc_index, const char *drc_name,

                                     void *fdt, int node_offset)

{

    int offset, ret;

    int slot = PCI_SLOT(dev->devfn);

    int func = PCI_FUNC(dev->devfn);

    char nodename[FDT_NAME_MAX];



    if (func != 0) {

        snprintf(nodename, FDT_NAME_MAX, "pci@%x,%x", slot, func);

    } else {

        snprintf(nodename, FDT_NAME_MAX, "pci@%x", slot);

    }

    offset = fdt_add_subnode(fdt, node_offset, nodename);

    ret = spapr_populate_pci_child_dt(dev, fdt, offset, phb->index, drc_index,

                                      phb);

    g_assert(!ret);

    if (ret) {

        return 0;

    }

    return offset;

}
