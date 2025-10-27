static void io_write(IPackDevice *ip, uint8_t addr, uint16_t val)

{

    IPOctalState *dev = IPOCTAL(ip);

    unsigned reg = val & 0xFF;

    /* addr[7:6]: block   (A-D)

       addr[7:5]: channel (a-h)

       addr[5:0]: register */

    unsigned block = addr >> 5;

    unsigned channel = addr >> 4;

    /* Big endian, accessed using 8-bit bytes at odd locations */

    unsigned offset = (addr & 0x1F) ^ 1;

    SCC2698Channel *ch = &dev->ch[channel];

    SCC2698Block *blk = &dev->blk[block];

    uint8_t old_isr = blk->isr;

    uint8_t old_imr = blk->imr;



    switch (offset) {



    case REG_MRa:

    case REG_MRb:

        ch->mr[ch->mr_idx] = reg;

        DPRINTF("Write MR%u%c 0x%x\n", ch->mr_idx + 1, channel + 'a', reg);

        ch->mr_idx = 1;

        break;



    /* Not implemented */

    case REG_CSRa:

    case REG_CSRb:

        DPRINTF("Write CSR%c: 0x%x\n", channel + 'a', reg);

        break;



    case REG_CRa:

    case REG_CRb:

        write_cr(dev, channel, reg);

        break;



    case REG_THRa:

    case REG_THRb:

        if (ch->sr & SR_TXRDY) {

            DPRINTF("Write THR%c (0x%x)\n", channel + 'a', reg);

            if (ch->dev) {

                uint8_t thr = reg;

                qemu_chr_fe_write(ch->dev, &thr, 1);

            }

        } else {

            DPRINTF("Write THR%c (0x%x), Tx disabled\n", channel + 'a', reg);

        }

        break;



    /* Not implemented */

    case REG_ACR:

        DPRINTF("Write ACR%c 0x%x\n", block + 'A', val);

        break;



    case REG_IMR:

        DPRINTF("Write IMR%c 0x%x\n", block + 'A', val);

        blk->imr = reg;

        break;



    /* Not implemented */

    case REG_OPCR:

        DPRINTF("Write OPCR%c 0x%x\n", block + 'A', val);

        break;



    default:

        DPRINTF("Write unknown/unsupported register 0x%02x %u\n", offset, val);

    }



    if (old_isr != blk->isr || old_imr != blk->imr) {

        update_irq(dev, block);

    }

}
