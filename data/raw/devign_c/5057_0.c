static uint64_t omap_eac_read(void *opaque, target_phys_addr_t addr,

                              unsigned size)

{

    struct omap_eac_s *s = (struct omap_eac_s *) opaque;

    uint32_t ret;



    if (size != 2) {

        return omap_badwidth_read16(opaque, addr);

    }



    switch (addr) {

    case 0x000:	/* CPCFR1 */

        return s->config[0];

    case 0x004:	/* CPCFR2 */

        return s->config[1];

    case 0x008:	/* CPCFR3 */

        return s->config[2];

    case 0x00c:	/* CPCFR4 */

        return s->config[3];



    case 0x010:	/* CPTCTL */

        return s->control | ((s->codec.rxavail + s->codec.rxlen > 0) << 7) |

                ((s->codec.txlen < s->codec.txavail) << 5);



    case 0x014:	/* CPTTADR */

        return s->address;

    case 0x018:	/* CPTDATL */

        return s->data & 0xff;

    case 0x01c:	/* CPTDATH */

        return s->data >> 8;

    case 0x020:	/* CPTVSLL */

        return s->vtol;

    case 0x024:	/* CPTVSLH */

        return s->vtsl | (3 << 5);	/* CRDY1 | CRDY2 */

    case 0x040:	/* MPCTR */

        return s->modem.control;

    case 0x044:	/* MPMCCFR */

        return s->modem.config;

    case 0x060:	/* BPCTR */

        return s->bt.control;

    case 0x064:	/* BPMCCFR */

        return s->bt.config;

    case 0x080:	/* AMSCFR */

        return s->mixer;

    case 0x084:	/* AMVCTR */

        return s->gain[0];

    case 0x088:	/* AM1VCTR */

        return s->gain[1];

    case 0x08c:	/* AM2VCTR */

        return s->gain[2];

    case 0x090:	/* AM3VCTR */

        return s->gain[3];

    case 0x094:	/* ASTCTR */

        return s->att;

    case 0x098:	/* APD1LCR */

        return s->max[0];

    case 0x09c:	/* APD1RCR */

        return s->max[1];

    case 0x0a0:	/* APD2LCR */

        return s->max[2];

    case 0x0a4:	/* APD2RCR */

        return s->max[3];

    case 0x0a8:	/* APD3LCR */

        return s->max[4];

    case 0x0ac:	/* APD3RCR */

        return s->max[5];

    case 0x0b0:	/* APD4R */

        return s->max[6];

    case 0x0b4:	/* ADWR */

        /* This should be write-only?  Docs list it as read-only.  */

        return 0x0000;

    case 0x0b8:	/* ADRDR */

        if (likely(s->codec.rxlen > 1)) {

            ret = s->codec.rxbuf[s->codec.rxoff ++];

            s->codec.rxlen --;

            s->codec.rxoff &= EAC_BUF_LEN - 1;

            return ret;

        } else if (s->codec.rxlen) {

            ret = s->codec.rxbuf[s->codec.rxoff ++];

            s->codec.rxlen --;

            s->codec.rxoff &= EAC_BUF_LEN - 1;

            if (s->codec.rxavail)

                omap_eac_in_refill(s);

            omap_eac_in_dmarequest_update(s);

            return ret;

        }

        return 0x0000;

    case 0x0bc:	/* AGCFR */

        return s->codec.config[0];

    case 0x0c0:	/* AGCTR */

        return s->codec.config[1] | ((s->codec.config[1] & 2) << 14);

    case 0x0c4:	/* AGCFR2 */

        return s->codec.config[2];

    case 0x0c8:	/* AGCFR3 */

        return s->codec.config[3];

    case 0x0cc:	/* MBPDMACTR */

    case 0x0d0:	/* MPDDMARR */

    case 0x0d8:	/* MPUDMARR */

    case 0x0e4:	/* BPDDMARR */

    case 0x0ec:	/* BPUDMARR */

        return 0x0000;



    case 0x100:	/* VERSION_NUMBER */

        return 0x0010;



    case 0x104:	/* SYSCONFIG */

        return s->sysconfig;



    case 0x108:	/* SYSSTATUS */

        return 1 | 0xe;					/* RESETDONE | stuff */

    }



    OMAP_BAD_REG(addr);

    return 0;

}
