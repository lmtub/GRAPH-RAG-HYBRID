static void l2x0_priv_write(void *opaque, target_phys_addr_t offset,

                            uint64_t value, unsigned size)

{

    l2x0_state *s = (l2x0_state *)opaque;

    offset &= 0xfff;

    if (offset >= 0x730 && offset < 0x800) {

        /* ignore */

        return;

    }

    switch (offset) {

    case 0x100:

        s->ctrl = value & 1;

        break;

    case 0x104:

        s->aux_ctrl = value;

        break;

    case 0x108:

        s->tag_ctrl = value;

        break;

    case 0x10C:

        s->data_ctrl = value;

        break;

    case 0xC00:

        s->filter_start = value;

        break;

    case 0xC04:

        s->filter_end = value;

        break;

    case 0xF40:

        return;

    case 0xF60:

        return;

    case 0xF80:

        return;

    default:

        fprintf(stderr, "l2x0_priv_write: Bad offset %x\n", (int)offset);

        break;

    }

}
