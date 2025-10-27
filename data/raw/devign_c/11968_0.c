static void imx_serial_write(void *opaque, target_phys_addr_t offset,

                      uint64_t value, unsigned size)

{

    IMXSerialState *s = (IMXSerialState *)opaque;

    unsigned char ch;



    DPRINTF("write(offset=%x, value = %x) to %s\n",

            offset >> 2,

            (unsigned int)value, s->chr ? s->chr->label : "NODEV");



    switch (offset >> 2) {

    case 0x10: /* UTXD */

        ch = value;

        if (s->ucr2 & UCR2_TXEN) {

            if (s->chr) {

                qemu_chr_fe_write(s->chr, &ch, 1);

            }

            s->usr1 &= ~USR1_TRDY;

            imx_update(s);

            s->usr1 |= USR1_TRDY;

            imx_update(s);

        }

        break;



    case 0x20: /* UCR1 */

        s->ucr1 = value & 0xffff;

        DPRINTF("write(ucr1=%x)\n", (unsigned int)value);

        imx_update(s);

        break;



    case 0x21: /* UCR2 */

        /*

         * Only a few bits in control register 2 are implemented as yet.

         * If it's intended to use a real serial device as a back-end, this

         * register will have to be implemented more fully.

         */

        if (!(value & UCR2_SRST)) {

            imx_serial_reset(s);

            imx_update(s);

            value |= UCR2_SRST;

        }

        if (value & UCR2_RXEN) {

            if (!(s->ucr2 & UCR2_RXEN)) {

                qemu_chr_accept_input(s->chr);

            }

        }

        s->ucr2 = value & 0xffff;

        break;



    case 0x25: /* USR1 */

        value &= USR1_AWAKE | USR1_AIRINT | USR1_DTRD | USR1_AGTIM |

            USR1_FRAMERR | USR1_ESCF | USR1_RTSD | USR1_PARTYER;

        s->usr1 &= ~value;

        break;



    case 0x26: /* USR2 */

       /*

        * Writing 1 to some bits clears them; all other

        * values are ignored

        */

        value &= USR2_ADET | USR2_DTRF | USR2_IDLE | USR2_ACST |

            USR2_RIDELT | USR2_IRINT | USR2_WAKE |

            USR2_DCDDELT | USR2_RTSF | USR2_BRCD | USR2_ORE;

        s->usr2 &= ~value;

        break;



        /*

         * Linux expects to see what it writes to these registers

         * We don't currently alter the baud rate

         */

    case 0x29: /* UBIR */

        s->ubrc = value & 0xffff;

        break;



    case 0x2a: /* UBMR */

        s->ubmr = value & 0xffff;

        break;



    case 0x2c: /* One ms reg */

        s->onems = value & 0xffff;

        break;



    case 0x24: /* FIFO control register */

        s->ufcr = value & 0xffff;

        break;



    case 0x22: /* UCR3 */

        s->ucr3 = value & 0xffff;

        break;



    case 0x2d: /* UTS1 */

    case 0x23: /* UCR4 */

        IPRINTF("Unimplemented Register %x written to\n", offset >> 2);

        /* TODO */

        break;



    default:

        IPRINTF("imx_serial_write: Bad offset 0x%x\n", (int)offset);

    }

}
