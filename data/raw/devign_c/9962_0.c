static uint64_t pcnet_ioport_read(void *opaque, target_phys_addr_t addr,

                                  unsigned size)

{

    PCNetState *d = opaque;



    if (addr < 16 && size == 1) {

        return pcnet_aprom_readb(d, addr);

    } else if (addr >= 0x10 && addr < 0x20 && size == 2) {

        return pcnet_ioport_readw(d, addr);

    } else if (addr >= 0x10 && addr < 0x20 && size == 4) {

        return pcnet_ioport_readl(d, addr);

    }

    return ((uint64_t)1 << (size * 8)) - 1;

}
