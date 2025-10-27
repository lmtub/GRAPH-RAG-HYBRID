static void pmac_ide_writeb (void *opaque,

                             target_phys_addr_t addr, uint32_t val)

{

    MACIOIDEState *d = opaque;



    addr = (addr & 0xFFF) >> 4;

    switch (addr) {

    case 1 ... 7:

        ide_ioport_write(&d->bus, addr, val);

        break;

    case 8:

    case 22:

        ide_cmd_write(&d->bus, 0, val);

        break;

    default:

        break;

    }

}
