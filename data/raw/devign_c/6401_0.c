static void a9_scu_write(void *opaque, target_phys_addr_t offset,

                         uint64_t value, unsigned size)

{

    a9mp_priv_state *s = (a9mp_priv_state *)opaque;

    uint32_t mask;

    uint32_t shift;

    switch (size) {

    case 1:

        mask = 0xff;

        break;

    case 2:

        mask = 0xffff;

        break;

    case 4:

        mask = 0xffffffff;

        break;

    default:

        fprintf(stderr, "Invalid size %u in write to a9 scu register %x\n",

                size, (unsigned)offset);

        return;

    }



    switch (offset) {

    case 0x00: /* Control */

        s->scu_control = value & 1;

        break;

    case 0x4: /* Configuration: RO */

        break;

    case 0x08: case 0x09: case 0x0A: case 0x0B: /* Power Control */

        shift = (offset - 0x8) * 8;

        s->scu_status &= ~(mask << shift);

        s->scu_status |= ((value & mask) << shift);

        break;

    case 0x0c: /* Invalidate All Registers In Secure State */

        /* no-op as we do not implement caches */

        break;

    case 0x40: /* Filtering Start Address Register */

    case 0x44: /* Filtering End Address Register */

        /* RAZ/WI, like an implementation with only one AXI master */

        break;

    case 0x50: /* SCU Access Control Register */

    case 0x54: /* SCU Non-secure Access Control Register */

        /* unimplemented, fall through */

    default:

        break;

    }

}
