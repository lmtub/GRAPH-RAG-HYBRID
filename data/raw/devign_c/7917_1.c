static void nand_command(NANDFlashState *s)

{

    unsigned int offset;

    switch (s->cmd) {

    case NAND_CMD_READ0:

        s->iolen = 0;

        break;



    case NAND_CMD_READID:

        s->ioaddr = s->io;

        s->iolen = 0;

        nand_pushio_byte(s, s->manf_id);

        nand_pushio_byte(s, s->chip_id);

        nand_pushio_byte(s, 'Q'); /* Don't-care byte (often 0xa5) */

        if (nand_flash_ids[s->chip_id].options & NAND_SAMSUNG_LP) {

            /* Page Size, Block Size, Spare Size; bit 6 indicates

             * 8 vs 16 bit width NAND.

             */

            nand_pushio_byte(s, (s->buswidth == 2) ? 0x55 : 0x15);

        } else {

            nand_pushio_byte(s, 0xc0); /* Multi-plane */

        }

        break;



    case NAND_CMD_RANDOMREAD2:

    case NAND_CMD_NOSERIALREAD2:

        if (!(nand_flash_ids[s->chip_id].options & NAND_SAMSUNG_LP))

            break;

        offset = s->addr & ((1 << s->addr_shift) - 1);

        s->blk_load(s, s->addr, offset);

        if (s->gnd)

            s->iolen = (1 << s->page_shift) - offset;

        else

            s->iolen = (1 << s->page_shift) + (1 << s->oob_shift) - offset;

        break;



    case NAND_CMD_RESET:

        nand_reset(&s->busdev.qdev);

        break;



    case NAND_CMD_PAGEPROGRAM1:

        s->ioaddr = s->io;

        s->iolen = 0;

        break;



    case NAND_CMD_PAGEPROGRAM2:

        if (s->wp) {

            s->blk_write(s);

        }

        break;



    case NAND_CMD_BLOCKERASE1:

        break;



    case NAND_CMD_BLOCKERASE2:

        s->addr &= (1ull << s->addrlen * 8) - 1;

        if (nand_flash_ids[s->chip_id].options & NAND_SAMSUNG_LP)

            s->addr <<= 16;

        else

            s->addr <<= 8;



        if (s->wp) {

            s->blk_erase(s);

        }

        break;



    case NAND_CMD_READSTATUS:

        s->ioaddr = s->io;

        s->iolen = 0;

        nand_pushio_byte(s, s->status);

        break;



    default:

        printf("%s: Unknown NAND command 0x%02x\n", __FUNCTION__, s->cmd);

    }

}
