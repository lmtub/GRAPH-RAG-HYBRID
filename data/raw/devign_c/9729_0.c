MemoryRegion *pci_address_space(PCIDevice *dev)

{

    return dev->bus->address_space_mem;

}
