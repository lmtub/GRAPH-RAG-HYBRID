static uint64_t openpic_src_read(void *opaque, uint64_t addr, unsigned len)

{

    OpenPICState *opp = opaque;

    uint32_t retval;

    int idx;



    DPRINTF("%s: addr %#" HWADDR_PRIx "\n", __func__, addr);

    retval = 0xFFFFFFFF;

    if (addr & 0xF) {

        return retval;

    }

    addr = addr & 0xFFF0;

    idx = addr >> 5;

    if (addr & 0x10) {

        /* EXDE / IFEDE / IEEDE */

        retval = read_IRQreg_idr(opp, idx);

    } else {

        /* EXVP / IFEVP / IEEVP */

        retval = read_IRQreg_ivpr(opp, idx);

    }

    DPRINTF("%s: => 0x%08x\n", __func__, retval);



    return retval;

}
