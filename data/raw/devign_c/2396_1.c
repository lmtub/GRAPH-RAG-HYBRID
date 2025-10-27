void *qpci_iomap(QPCIDevice *dev, int barno, uint64_t *sizeptr)

{

    QPCIBus *bus = dev->bus;

    static const int bar_reg_map[] = {

        PCI_BASE_ADDRESS_0, PCI_BASE_ADDRESS_1, PCI_BASE_ADDRESS_2,

        PCI_BASE_ADDRESS_3, PCI_BASE_ADDRESS_4, PCI_BASE_ADDRESS_5,

    };

    int bar_reg;

    uint32_t addr, size;

    uint32_t io_type;

    uint64_t loc;



    g_assert(barno >= 0 && barno <= 5);

    bar_reg = bar_reg_map[barno];



    qpci_config_writel(dev, bar_reg, 0xFFFFFFFF);

    addr = qpci_config_readl(dev, bar_reg);



    io_type = addr & PCI_BASE_ADDRESS_SPACE;

    if (io_type == PCI_BASE_ADDRESS_SPACE_IO) {

        addr &= PCI_BASE_ADDRESS_IO_MASK;

    } else {

        addr &= PCI_BASE_ADDRESS_MEM_MASK;

    }



    g_assert(addr); /* Must have *some* size bits */



    size = 1U << ctz32(addr);

    if (sizeptr) {

        *sizeptr = size;

    }



    if (io_type == PCI_BASE_ADDRESS_SPACE_IO) {

        loc = QEMU_ALIGN_UP(bus->pio_alloc_ptr, size);



        g_assert(loc >= bus->pio_alloc_ptr);

        g_assert(loc + size <= QPCI_PIO_LIMIT); /* Keep PIO below 64kiB */



        bus->pio_alloc_ptr = loc + size;



        qpci_config_writel(dev, bar_reg, loc | PCI_BASE_ADDRESS_SPACE_IO);

    } else {

        loc = QEMU_ALIGN_UP(bus->mmio_alloc_ptr, size);



        /* Check for space */

        g_assert(loc >= bus->mmio_alloc_ptr);

        g_assert(loc + size <= bus->mmio_limit);



        bus->mmio_alloc_ptr = loc + size;



        qpci_config_writel(dev, bar_reg, loc);

    }



    return (void *)(uintptr_t)loc;

}
