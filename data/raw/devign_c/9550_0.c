static void omap_mcbsp_writew(void *opaque, hwaddr addr,

                uint32_t value)

{

    struct omap_mcbsp_s *s = (struct omap_mcbsp_s *) opaque;

    int offset = addr & OMAP_MPUI_REG_MASK;



    if (offset == 0x04) {				/* DXR */

        if (((s->xcr[0] >> 5) & 7) < 3)			/* XWDLEN1 */

            return;

        if (s->tx_req > 3) {

            s->tx_req -= 4;

            if (s->codec && s->codec->cts) {

                s->codec->out.fifo[s->codec->out.len ++] =

                        (value >> 24) & 0xff;

                s->codec->out.fifo[s->codec->out.len ++] =

                        (value >> 16) & 0xff;

                s->codec->out.fifo[s->codec->out.len ++] =

                        (value >> 8) & 0xff;

                s->codec->out.fifo[s->codec->out.len ++] =

                        (value >> 0) & 0xff;

            }

            if (s->tx_req < 4)

                omap_mcbsp_tx_done(s);

        } else

            printf("%s: Tx FIFO overrun\n", __FUNCTION__);

        return;

    }



    omap_badwidth_write16(opaque, addr, value);

}
