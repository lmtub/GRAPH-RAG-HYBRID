static void qpci_spapr_io_writeb(QPCIBus *bus, void *addr, uint8_t value)

{

    QPCIBusSPAPR *s = container_of(bus, QPCIBusSPAPR, bus);

    uint64_t port = (uintptr_t)addr;

    if (port < s->pio.size) {

        writeb(s->pio_cpu_base + port, value);

    } else {

        writeb(s->mmio_cpu_base + port, value);

    }

}
