void pci_register_bar(PCIDevice *pci_dev, int region_num,

                      uint8_t type, MemoryRegion *memory)

{

    PCIIORegion *r;

    uint32_t addr; /* offset in pci config space */

    uint64_t wmask;

    pcibus_t size = memory_region_size(memory);



    assert(region_num >= 0);

    assert(region_num < PCI_NUM_REGIONS);

    if (size & (size-1)) {

        fprintf(stderr, "ERROR: PCI region size must be pow2 "

                    "type=0x%x, size=0x%"FMT_PCIBUS"\n", type, size);

        exit(1);

    }



    r = &pci_dev->io_regions[region_num];

    r->addr = PCI_BAR_UNMAPPED;

    r->size = size;

    r->type = type;

    r->memory = memory;

    r->address_space = type & PCI_BASE_ADDRESS_SPACE_IO

                        ? pci_dev->bus->address_space_io

                        : pci_dev->bus->address_space_mem;



    wmask = ~(size - 1);

    if (region_num == PCI_ROM_SLOT) {

        /* ROM enable bit is writable */

        wmask |= PCI_ROM_ADDRESS_ENABLE;

    }



    addr = pci_bar(pci_dev, region_num);

    pci_set_long(pci_dev->config + addr, type);



    if (!(r->type & PCI_BASE_ADDRESS_SPACE_IO) &&

        r->type & PCI_BASE_ADDRESS_MEM_TYPE_64) {

        pci_set_quad(pci_dev->wmask + addr, wmask);

        pci_set_quad(pci_dev->cmask + addr, ~0ULL);

    } else {

        pci_set_long(pci_dev->wmask + addr, wmask & 0xffffffff);

        pci_set_long(pci_dev->cmask + addr, 0xffffffff);

    }

}
