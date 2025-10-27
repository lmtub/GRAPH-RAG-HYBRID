static uint64_t mmio_ide_read(void *opaque, target_phys_addr_t addr,

                              unsigned size)

{

    MMIOState *s = opaque;

    addr >>= s->shift;

    if (addr & 7)

        return ide_ioport_read(&s->bus, addr);

    else

        return ide_data_readw(&s->bus, 0);

}
