static uint64_t pxa2xx_i2s_read(void *opaque, hwaddr addr,

                                unsigned size)

{

    PXA2xxI2SState *s = (PXA2xxI2SState *) opaque;



    switch (addr) {

    case SACR0:

        return s->control[0];

    case SACR1:

        return s->control[1];

    case SASR0:

        return s->status;

    case SAIMR:

        return s->mask;

    case SAICR:

        return 0;

    case SADIV:

        return s->clk;

    case SADR:

        if (s->rx_len > 0) {

            s->rx_len --;

            pxa2xx_i2s_update(s);

            return s->codec_in(s->opaque);

        }

        return 0;

    default:

        printf("%s: Bad register " REG_FMT "\n", __FUNCTION__, addr);

        break;

    }

    return 0;

}
