static void omap_uwire_write(void *opaque, target_phys_addr_t addr,

                             uint64_t value, unsigned size)

{

    struct omap_uwire_s *s = (struct omap_uwire_s *) opaque;

    int offset = addr & OMAP_MPUI_REG_MASK;



    if (size != 2) {

        return omap_badwidth_write16(opaque, addr, value);

    }



    switch (offset) {

    case 0x00:	/* TDR */

        s->txbuf = value;				/* TD */

        if ((s->setup[4] & (1 << 2)) &&			/* AUTO_TX_EN */

                        ((s->setup[4] & (1 << 3)) ||	/* CS_TOGGLE_TX_EN */

                         (s->control & (1 << 12)))) {	/* CS_CMD */

            s->control |= 1 << 14;			/* CSRB */

            omap_uwire_transfer_start(s);

        }

        break;



    case 0x04:	/* CSR */

        s->control = value & 0x1fff;

        if (value & (1 << 13))				/* START */

            omap_uwire_transfer_start(s);

        break;



    case 0x08:	/* SR1 */

        s->setup[0] = value & 0x003f;

        break;



    case 0x0c:	/* SR2 */

        s->setup[1] = value & 0x0fc0;

        break;



    case 0x10:	/* SR3 */

        s->setup[2] = value & 0x0003;

        break;



    case 0x14:	/* SR4 */

        s->setup[3] = value & 0x0001;

        break;



    case 0x18:	/* SR5 */

        s->setup[4] = value & 0x000f;

        break;



    default:

        OMAP_BAD_REG(addr);

        return;

    }

}
