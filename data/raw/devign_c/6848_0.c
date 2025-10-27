static void openpic_src_write(void *opaque, hwaddr addr, uint64_t val,

                              unsigned len)

{

    OpenPICState *opp = opaque;

    int idx;



    DPRINTF("%s: addr %#" HWADDR_PRIx " <= %08" PRIx64 "\n",

            __func__, addr, val);

    if (addr & 0xF) {

        return;

    }

    addr = addr & 0xFFF0;

    idx = addr >> 5;

    if (addr & 0x10) {

        /* EXDE / IFEDE / IEEDE */

        write_IRQreg_idr(opp, idx, val);

    } else {

        /* EXVP / IFEVP / IEEVP */

        write_IRQreg_ivpr(opp, idx, val);

    }

}
