static uint64_t openpic_msi_read(void *opaque, hwaddr addr, unsigned size)

{

    OpenPICState *opp = opaque;

    uint64_t r = 0;

    int i, srs;



    DPRINTF("%s: addr " TARGET_FMT_plx "\n", __func__, addr);

    if (addr & 0xF) {

        return -1;

    }



    srs = addr >> 4;



    switch (addr) {

    case 0x00:

    case 0x10:

    case 0x20:

    case 0x30:

    case 0x40:

    case 0x50:

    case 0x60:

    case 0x70: /* MSIRs */

        r = opp->msi[srs].msir;

        /* Clear on read */

        opp->msi[srs].msir = 0;

        break;

    case 0x120: /* MSISR */

        for (i = 0; i < MAX_MSI; i++) {

            r |= (opp->msi[i].msir ? 1 : 0) << i;

        }

        break;

    }



    return r;

}
