static void tcx_blit_writel(void *opaque, hwaddr addr,

                            uint64_t val, unsigned size)

{

    TCXState *s = opaque;

    uint32_t adsr, len;

    int i;



    if (!(addr & 4)) {

        s->tmpblit = val;

    } else {

        addr = (addr >> 3) & 0xfffff;

        adsr = val & 0xffffff;

        len = ((val >> 24) & 0x1f) + 1;

        if (adsr == 0xffffff) {

            memset(&s->vram[addr], s->tmpblit, len);

            if (s->depth == 24) {

                val = s->tmpblit & 0xffffff;

                val = cpu_to_be32(val);

                for (i = 0; i < len; i++) {

                    s->vram24[addr + i] = val;

                }

            }

        } else {

            memcpy(&s->vram[addr], &s->vram[adsr], len);

            if (s->depth == 24) {

                memcpy(&s->vram24[addr], &s->vram24[adsr], len * 4);

            }

        }

        memory_region_set_dirty(&s->vram_mem, addr, len);

    }

}
