static void mmio_ide_write(void *opaque, target_phys_addr_t addr,

                           uint64_t val, unsigned size)

{

    MMIOState *s = opaque;

    addr >>= s->shift;

    if (addr & 7)

        ide_ioport_write(&s->bus, addr, val);

    else

        ide_data_writew(&s->bus, 0, val);

}
