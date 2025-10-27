static uint64_t openpic_tmr_read(void *opaque, hwaddr addr, unsigned len)

{

    OpenPICState *opp = opaque;

    uint32_t retval = -1;

    int idx;



    DPRINTF("%s: addr %08x\n", __func__, addr);

    if (addr & 0xF) {

        goto out;

    }

    idx = (addr >> 6) & 0x3;

    if (addr == 0x0) {

        /* TIFR (TFRR) */

        retval = opp->tifr;

        goto out;

    }

    switch (addr & 0x30) {

    case 0x00: /* TICC (GTCCR) */

        retval = opp->timers[idx].ticc;

        break;

    case 0x10: /* TIBC (GTBCR) */

        retval = opp->timers[idx].tibc;

        break;

    case 0x20: /* TIPV (TIPV) */

        retval = read_IRQreg_ipvp(opp, opp->irq_tim0 + idx);

        break;

    case 0x30: /* TIDE (TIDR) */

        retval = read_IRQreg_ide(opp, opp->irq_tim0 + idx);

        break;

    }



out:

    DPRINTF("%s: => %08x\n", __func__, retval);



    return retval;

}
