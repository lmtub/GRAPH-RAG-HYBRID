static uint32_t qpci_spapr_io_readl(QPCIBus *bus, void *addr)

{

    QPCIBusSPAPR *s = container_of(bus, QPCIBusSPAPR, bus);

    uint64_t port = (uintptr_t)addr;

    uint32_t v;

    if (port < s->pio.size) {

        v = readl(s->pio_cpu_base + port);

    } else {

        v = readl(s->mmio_cpu_base + port);

    }

    return bswap32(v);

}
