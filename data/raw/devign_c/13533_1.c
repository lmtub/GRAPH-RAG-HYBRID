static void ebus_mmio_mapfunc(PCIDevice *pci_dev, int region_num,

                              pcibus_t addr, pcibus_t size, int type)

{

    EBUS_DPRINTF("Mapping region %d registers at %" FMT_PCIBUS "\n",

                 region_num, addr);

    switch (region_num) {

    case 0:

        isa_mmio_init(addr, 0x1000000);

        break;

    case 1:

        isa_mmio_init(addr, 0x800000);

        break;

    }

}
