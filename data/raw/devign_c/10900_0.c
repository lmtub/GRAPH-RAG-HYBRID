static PXBDev *convert_to_pxb(PCIDevice *dev)

{

    return pci_bus_is_express(dev->bus) ? PXB_PCIE_DEV(dev) : PXB_DEV(dev);

}
