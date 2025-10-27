static uint64_t l2x0_priv_read(void *opaque, target_phys_addr_t offset,

                               unsigned size)

{

    uint32_t cache_data;

    l2x0_state *s = (l2x0_state *)opaque;

    offset &= 0xfff;

    if (offset >= 0x730 && offset < 0x800) {

        return 0; /* cache ops complete */

    }

    switch (offset) {

    case 0:

        return CACHE_ID;

    case 0x4:

        /* aux_ctrl values affect cache_type values */

        cache_data = (s->aux_ctrl & (7 << 17)) >> 15;

        cache_data |= (s->aux_ctrl & (1 << 16)) >> 16;

        return s->cache_type |= (cache_data << 18) | (cache_data << 6);

    case 0x100:

        return s->ctrl;

    case 0x104:

        return s->aux_ctrl;

    case 0x108:

        return s->tag_ctrl;

    case 0x10C:

        return s->data_ctrl;

    case 0xC00:

        return s->filter_start;

    case 0xC04:

        return s->filter_end;

    case 0xF40:

        return 0;

    case 0xF60:

        return 0;

    case 0xF80:

        return 0;

    default:

        fprintf(stderr, "l2x0_priv_read: Bad offset %x\n", (int)offset);

        break;

    }

    return 0;

}
