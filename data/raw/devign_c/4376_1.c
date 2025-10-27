static void imx_eth_write(void *opaque, hwaddr offset, uint64_t value,

                           unsigned size)

{

    IMXFECState *s = IMX_FEC(opaque);

    uint32_t index = offset >> 2;



    FEC_PRINTF("reg[%s] <= 0x%" PRIx32 "\n", imx_eth_reg_name(s, index),

                (uint32_t)value);



    switch (index) {

    case ENET_EIR:

        s->regs[index] &= ~value;

        break;

    case ENET_EIMR:

        s->regs[index] = value;

        break;

    case ENET_RDAR:

        if (s->regs[ENET_ECR] & ENET_ECR_ETHEREN) {

            if (!s->regs[index]) {

                s->regs[index] = ENET_RDAR_RDAR;

                imx_eth_enable_rx(s);

            }

        } else {

            s->regs[index] = 0;

        }

        break;

    case ENET_TDAR:

        if (s->regs[ENET_ECR] & ENET_ECR_ETHEREN) {

            s->regs[index] = ENET_TDAR_TDAR;

            imx_eth_do_tx(s);

        }

        s->regs[index] = 0;

        break;

    case ENET_ECR:

        if (value & ENET_ECR_RESET) {

            return imx_eth_reset(DEVICE(s));

        }

        s->regs[index] = value;

        if ((s->regs[index] & ENET_ECR_ETHEREN) == 0) {

            s->regs[ENET_RDAR] = 0;

            s->rx_descriptor = s->regs[ENET_RDSR];

            s->regs[ENET_TDAR] = 0;

            s->tx_descriptor = s->regs[ENET_TDSR];

        }

        break;

    case ENET_MMFR:

        s->regs[index] = value;

        if (extract32(value, 29, 1)) {

            /* This is a read operation */

            s->regs[ENET_MMFR] = deposit32(s->regs[ENET_MMFR], 0, 16,

                                           do_phy_read(s,

                                                       extract32(value,

                                                                 18, 10)));

        } else {

            /* This a write operation */

            do_phy_write(s, extract32(value, 18, 10), extract32(value, 0, 16));

        }

        /* raise the interrupt as the PHY operation is done */

        s->regs[ENET_EIR] |= ENET_INT_MII;

        break;

    case ENET_MSCR:

        s->regs[index] = value & 0xfe;

        break;

    case ENET_MIBC:

        /* TODO: Implement MIB.  */

        s->regs[index] = (value & 0x80000000) ? 0xc0000000 : 0;

        break;

    case ENET_RCR:

        s->regs[index] = value & 0x07ff003f;

        /* TODO: Implement LOOP mode.  */

        break;

    case ENET_TCR:

        /* We transmit immediately, so raise GRA immediately.  */

        s->regs[index] = value;

        if (value & 1) {

            s->regs[ENET_EIR] |= ENET_INT_GRA;

        }

        break;

    case ENET_PALR:

        s->regs[index] = value;

        s->conf.macaddr.a[0] = value >> 24;

        s->conf.macaddr.a[1] = value >> 16;

        s->conf.macaddr.a[2] = value >> 8;

        s->conf.macaddr.a[3] = value;

        break;

    case ENET_PAUR:

        s->regs[index] = (value | 0x0000ffff) & 0xffff8808;

        s->conf.macaddr.a[4] = value >> 24;

        s->conf.macaddr.a[5] = value >> 16;

        break;

    case ENET_OPD:

        s->regs[index] = (value & 0x0000ffff) | 0x00010000;

        break;

    case ENET_IAUR:

    case ENET_IALR:

    case ENET_GAUR:

    case ENET_GALR:

        /* TODO: implement MAC hash filtering.  */

        break;

    case ENET_TFWR:

        if (s->is_fec) {

            s->regs[index] = value & 0x3;

        } else {

            s->regs[index] = value & 0x13f;

        }

        break;

    case ENET_RDSR:

        if (s->is_fec) {

            s->regs[index] = value & ~3;

        } else {

            s->regs[index] = value & ~7;

        }

        s->rx_descriptor = s->regs[index];

        break;

    case ENET_TDSR:

        if (s->is_fec) {

            s->regs[index] = value & ~3;

        } else {

            s->regs[index] = value & ~7;

        }

        s->tx_descriptor = s->regs[index];

        break;

    case ENET_MRBR:

        s->regs[index] = value & 0x00003ff0;

        break;

    default:

        if (s->is_fec) {

            imx_fec_write(s, index, value);

        } else {

            imx_enet_write(s, index, value);

        }

        return;

    }



    imx_eth_update(s);

}
