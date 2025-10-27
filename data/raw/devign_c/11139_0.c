static uint64_t pxa2xx_ssp_read(void *opaque, hwaddr addr,

                                unsigned size)

{

    PXA2xxSSPState *s = (PXA2xxSSPState *) opaque;

    uint32_t retval;



    switch (addr) {

    case SSCR0:

        return s->sscr[0];

    case SSCR1:

        return s->sscr[1];

    case SSPSP:

        return s->sspsp;

    case SSTO:

        return s->ssto;

    case SSITR:

        return s->ssitr;

    case SSSR:

        return s->sssr | s->ssitr;

    case SSDR:

        if (!s->enable)

            return 0xffffffff;

        if (s->rx_level < 1) {

            printf("%s: SSP Rx Underrun\n", __FUNCTION__);

            return 0xffffffff;

        }

        s->rx_level --;

        retval = s->rx_fifo[s->rx_start ++];

        s->rx_start &= 0xf;

        pxa2xx_ssp_fifo_update(s);

        return retval;

    case SSTSA:

        return s->sstsa;

    case SSRSA:

        return s->ssrsa;

    case SSTSS:

        return 0;

    case SSACD:

        return s->ssacd;

    default:

        printf("%s: Bad register " REG_FMT "\n", __FUNCTION__, addr);

        break;

    }

    return 0;

}
