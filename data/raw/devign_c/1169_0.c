static void pxa2xx_ssp_write(void *opaque, hwaddr addr,

                             uint64_t value64, unsigned size)

{

    PXA2xxSSPState *s = (PXA2xxSSPState *) opaque;

    uint32_t value = value64;



    switch (addr) {

    case SSCR0:

        s->sscr[0] = value & 0xc7ffffff;

        s->enable = value & SSCR0_SSE;

        if (value & SSCR0_MOD)

            printf("%s: Attempt to use network mode\n", __FUNCTION__);

        if (s->enable && SSCR0_DSS(value) < 4)

            printf("%s: Wrong data size: %i bits\n", __FUNCTION__,

                            SSCR0_DSS(value));

        if (!(value & SSCR0_SSE)) {

            s->sssr = 0;

            s->ssitr = 0;

            s->rx_level = 0;

        }

        pxa2xx_ssp_fifo_update(s);

        break;



    case SSCR1:

        s->sscr[1] = value;

        if (value & (SSCR1_LBM | SSCR1_EFWR))

            printf("%s: Attempt to use SSP test mode\n", __FUNCTION__);

        pxa2xx_ssp_fifo_update(s);

        break;



    case SSPSP:

        s->sspsp = value;

        break;



    case SSTO:

        s->ssto = value;

        break;



    case SSITR:

        s->ssitr = value & SSITR_INT;

        pxa2xx_ssp_int_update(s);

        break;



    case SSSR:

        s->sssr &= ~(value & SSSR_RW);

        pxa2xx_ssp_int_update(s);

        break;



    case SSDR:

        if (SSCR0_UWIRE(s->sscr[0])) {

            if (s->sscr[1] & SSCR1_MWDS)

                value &= 0xffff;

            else

                value &= 0xff;

        } else

            /* Note how 32bits overflow does no harm here */

            value &= (1 << SSCR0_DSS(s->sscr[0])) - 1;



        /* Data goes from here to the Tx FIFO and is shifted out from

         * there directly to the slave, no need to buffer it.

         */

        if (s->enable) {

            uint32_t readval;

            readval = ssi_transfer(s->bus, value);

            if (s->rx_level < 0x10) {

                s->rx_fifo[(s->rx_start + s->rx_level ++) & 0xf] = readval;

            } else {

                s->sssr |= SSSR_ROR;

            }

        }

        pxa2xx_ssp_fifo_update(s);

        break;



    case SSTSA:

        s->sstsa = value;

        break;



    case SSRSA:

        s->ssrsa = value;

        break;



    case SSACD:

        s->ssacd = value;

        break;



    default:

        printf("%s: Bad register " REG_FMT "\n", __FUNCTION__, addr);

        break;

    }

}
