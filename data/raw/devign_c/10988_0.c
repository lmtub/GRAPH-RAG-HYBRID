static uint32_t pflash_read (pflash_t *pfl, target_phys_addr_t offset,

                             int width, int be)

{

    target_phys_addr_t boff;

    uint32_t ret;

    uint8_t *p;



    DPRINTF("%s: offset " TARGET_FMT_plx "\n", __func__, offset);

    ret = -1;

    if (pfl->rom_mode) {

        /* Lazy reset of to ROMD mode */

        if (pfl->wcycle == 0)

            pflash_register_memory(pfl, 1);

    }

    offset &= pfl->chip_len - 1;

    boff = offset & 0xFF;

    if (pfl->width == 2)

        boff = boff >> 1;

    else if (pfl->width == 4)

        boff = boff >> 2;

    switch (pfl->cmd) {

    default:

        /* This should never happen : reset state & treat it as a read*/

        DPRINTF("%s: unknown command state: %x\n", __func__, pfl->cmd);

        pfl->wcycle = 0;

        pfl->cmd = 0;

    case 0x80:

        /* We accept reads during second unlock sequence... */

    case 0x00:

    flash_read:

        /* Flash area read */

        p = pfl->storage;

        switch (width) {

        case 1:

            ret = p[offset];

//            DPRINTF("%s: data offset %08x %02x\n", __func__, offset, ret);

            break;

        case 2:

            if (be) {

                ret = p[offset] << 8;

                ret |= p[offset + 1];

            } else {

                ret = p[offset];

                ret |= p[offset + 1] << 8;

            }

//            DPRINTF("%s: data offset %08x %04x\n", __func__, offset, ret);

            break;

        case 4:

            if (be) {

                ret = p[offset] << 24;

                ret |= p[offset + 1] << 16;

                ret |= p[offset + 2] << 8;

                ret |= p[offset + 3];

            } else {

                ret = p[offset];

                ret |= p[offset + 1] << 8;

                ret |= p[offset + 2] << 16;

                ret |= p[offset + 3] << 24;

            }

//            DPRINTF("%s: data offset %08x %08x\n", __func__, offset, ret);

            break;

        }

        break;

    case 0x90:

        /* flash ID read */

        switch (boff) {

        case 0x00:

        case 0x01:

            ret = pfl->ident[boff & 0x01];

            break;

        case 0x02:

            ret = 0x00; /* Pretend all sectors are unprotected */

            break;

        case 0x0E:

        case 0x0F:

            if (pfl->ident[2 + (boff & 0x01)] == (uint8_t)-1)

                goto flash_read;

            ret = pfl->ident[2 + (boff & 0x01)];

            break;

        default:

            goto flash_read;

        }

        DPRINTF("%s: ID " TARGET_FMT_pld " %x\n", __func__, boff, ret);

        break;

    case 0xA0:

    case 0x10:

    case 0x30:

        /* Status register read */

        ret = pfl->status;

        DPRINTF("%s: status %x\n", __func__, ret);

        /* Toggle bit 6 */

        pfl->status ^= 0x40;

        break;

    case 0x98:

        /* CFI query mode */

        if (boff > pfl->cfi_len)

            ret = 0;

        else

            ret = pfl->cfi_table[boff];

        break;

    }



    return ret;

}
