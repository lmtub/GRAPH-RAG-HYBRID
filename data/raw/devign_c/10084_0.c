static uint16_t md_common_read(PCMCIACardState *card, uint32_t at)

{

    MicroDriveState *s = MICRODRIVE(card);

    IDEState *ifs;

    uint16_t ret;

    at -= s->io_base;



    switch (s->opt & OPT_MODE) {

    case OPT_MODE_MMAP:

        if ((at & ~0x3ff) == 0x400) {

            at = 0;

        }

        break;

    case OPT_MODE_IOMAP16:

        at &= 0xf;

        break;

    case OPT_MODE_IOMAP1:

        if ((at & ~0xf) == 0x3f0) {

            at -= 0x3e8;

        } else if ((at & ~0xf) == 0x1f0) {

            at -= 0x1f0;

        }

        break;

    case OPT_MODE_IOMAP2:

        if ((at & ~0xf) == 0x370) {

            at -= 0x368;

        } else if ((at & ~0xf) == 0x170) {

            at -= 0x170;

        }

    }



    switch (at) {

    case 0x0:	/* Even RD Data */

    case 0x8:

        return ide_data_readw(&s->bus, 0);



        /* TODO: 8-bit accesses */

        if (s->cycle) {

            ret = s->io >> 8;

        } else {

            s->io = ide_data_readw(&s->bus, 0);

            ret = s->io & 0xff;

        }

        s->cycle = !s->cycle;

        return ret;

    case 0x9:	/* Odd RD Data */

        return s->io >> 8;

    case 0xd:	/* Error */

        return ide_ioport_read(&s->bus, 0x1);

    case 0xe:	/* Alternate Status */

        ifs = idebus_active_if(&s->bus);

        if (ifs->bs) {

            return ifs->status;

        } else {

            return 0;

        }

    case 0xf:	/* Device Address */

        ifs = idebus_active_if(&s->bus);

        return 0xc2 | ((~ifs->select << 2) & 0x3c);

    default:

        return ide_ioport_read(&s->bus, at);

    }



    return 0;

}
