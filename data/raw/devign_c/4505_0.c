static void tcx_rstip_writel(void *opaque, hwaddr addr,

                             uint64_t val, unsigned size)

{

    TCXState *s = opaque;

    int i;

    uint32_t col;



    if (!(addr & 4)) {

        s->tmpblit = val;

    } else {

        addr = (addr >> 3) & 0xfffff;

        col = cpu_to_be32(s->tmpblit);

        if (s->depth == 24) {

            for (i = 0; i < 32; i++) {

                if (val & 0x80000000) {

                    s->vram[addr + i] = s->tmpblit;

                    s->vram24[addr + i] = col;

                    s->cplane[addr + i] = col;

                }

                val <<= 1;

            }

        } else {

            for (i = 0; i < 32; i++)  {

                if (val & 0x80000000) {

                    s->vram[addr + i] = s->tmpblit;

                }

                val <<= 1;

            }

        }

        memory_region_set_dirty(&s->vram_mem, addr, 32);

    }

}
