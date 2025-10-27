static uint64_t omap_mcbsp_read(void *opaque, hwaddr addr,

                                unsigned size)

{

    struct omap_mcbsp_s *s = (struct omap_mcbsp_s *) opaque;

    int offset = addr & OMAP_MPUI_REG_MASK;

    uint16_t ret;



    if (size != 2) {

        return omap_badwidth_read16(opaque, addr);

    }



    switch (offset) {

    case 0x00:	/* DRR2 */

        if (((s->rcr[0] >> 5) & 7) < 3)			/* RWDLEN1 */

            return 0x0000;

        /* Fall through.  */

    case 0x02:	/* DRR1 */

        if (s->rx_req < 2) {

            printf("%s: Rx FIFO underrun\n", __FUNCTION__);

            omap_mcbsp_rx_done(s);

        } else {

            s->tx_req -= 2;

            if (s->codec && s->codec->in.len >= 2) {

                ret = s->codec->in.fifo[s->codec->in.start ++] << 8;

                ret |= s->codec->in.fifo[s->codec->in.start ++];

                s->codec->in.len -= 2;

            } else

                ret = 0x0000;

            if (!s->tx_req)

                omap_mcbsp_rx_done(s);

            return ret;

        }

        return 0x0000;



    case 0x04:	/* DXR2 */

    case 0x06:	/* DXR1 */

        return 0x0000;



    case 0x08:	/* SPCR2 */

        return s->spcr[1];

    case 0x0a:	/* SPCR1 */

        return s->spcr[0];

    case 0x0c:	/* RCR2 */

        return s->rcr[1];

    case 0x0e:	/* RCR1 */

        return s->rcr[0];

    case 0x10:	/* XCR2 */

        return s->xcr[1];

    case 0x12:	/* XCR1 */

        return s->xcr[0];

    case 0x14:	/* SRGR2 */

        return s->srgr[1];

    case 0x16:	/* SRGR1 */

        return s->srgr[0];

    case 0x18:	/* MCR2 */

        return s->mcr[1];

    case 0x1a:	/* MCR1 */

        return s->mcr[0];

    case 0x1c:	/* RCERA */

        return s->rcer[0];

    case 0x1e:	/* RCERB */

        return s->rcer[1];

    case 0x20:	/* XCERA */

        return s->xcer[0];

    case 0x22:	/* XCERB */

        return s->xcer[1];

    case 0x24:	/* PCR0 */

        return s->pcr;

    case 0x26:	/* RCERC */

        return s->rcer[2];

    case 0x28:	/* RCERD */

        return s->rcer[3];

    case 0x2a:	/* XCERC */

        return s->xcer[2];

    case 0x2c:	/* XCERD */

        return s->xcer[3];

    case 0x2e:	/* RCERE */

        return s->rcer[4];

    case 0x30:	/* RCERF */

        return s->rcer[5];

    case 0x32:	/* XCERE */

        return s->xcer[4];

    case 0x34:	/* XCERF */

        return s->xcer[5];

    case 0x36:	/* RCERG */

        return s->rcer[6];

    case 0x38:	/* RCERH */

        return s->rcer[7];

    case 0x3a:	/* XCERG */

        return s->xcer[6];

    case 0x3c:	/* XCERH */

        return s->xcer[7];

    }



    OMAP_BAD_REG(addr);

    return 0;

}
