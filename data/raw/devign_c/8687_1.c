e1000_mmio_map(PCIDevice *pci_dev, int region_num,
                uint32_t addr, uint32_t size, int type)
{
    E1000State *d = (E1000State *)pci_dev;
    DBGOUT(MMIO, "e1000_mmio_map addr=0x%08x 0x%08x\n", addr, size);
    cpu_register_physical_memory(addr, PNPMMIO_SIZE, d->mmio_index);
}