static void rc4030_write(void *opaque, hwaddr addr, uint64_t data,

                         unsigned int size)

{

    rc4030State *s = opaque;

    uint32_t val = data;

    addr &= 0x3fff;



    trace_rc4030_write(addr, val);



    switch (addr & ~0x3) {

    /* Global config register */

    case 0x0000:

        s->config = val;

        break;

    /* DMA transl. table base */

    case 0x0018:

        rc4030_dma_tt_update(s, val, s->dma_tl_limit);

        break;

    /* DMA transl. table limit */

    case 0x0020:

        rc4030_dma_tt_update(s, s->dma_tl_base, val);

        break;

    /* DMA transl. table invalidated */

    case 0x0028:

        break;

    /* Cache Maintenance */

    case 0x0030:

        s->cache_maint = val;

        break;

    /* I/O Cache Physical Tag */

    case 0x0048:

        s->cache_ptag = val;

        break;

    /* I/O Cache Logical Tag */

    case 0x0050:

        s->cache_ltag = val;

        break;

    /* I/O Cache Byte Mask */

    case 0x0058:

        s->cache_bmask |= val; /* HACK */

        break;

    /* I/O Cache Buffer Window */

    case 0x0060:

        /* HACK */

        if (s->cache_ltag == 0x80000001 && s->cache_bmask == 0xf0f0f0f) {

            hwaddr dest = s->cache_ptag & ~0x1;

            dest += (s->cache_maint & 0x3) << 3;

            cpu_physical_memory_write(dest, &val, 4);

        }

        break;

    /* Remote Speed Registers */

    case 0x0070:

    case 0x0078:

    case 0x0080:

    case 0x0088:

    case 0x0090:

    case 0x0098:

    case 0x00a0:

    case 0x00a8:

    case 0x00b0:

    case 0x00b8:

    case 0x00c0:

    case 0x00c8:

    case 0x00d0:

    case 0x00d8:

    case 0x00e0:

    case 0x00e8:

        s->rem_speed[(addr - 0x0070) >> 3] = val;

        break;

    /* DMA channel base address */

    case 0x0100:

    case 0x0108:

    case 0x0110:

    case 0x0118:

    case 0x0120:

    case 0x0128:

    case 0x0130:

    case 0x0138:

    case 0x0140:

    case 0x0148:

    case 0x0150:

    case 0x0158:

    case 0x0160:

    case 0x0168:

    case 0x0170:

    case 0x0178:

    case 0x0180:

    case 0x0188:

    case 0x0190:

    case 0x0198:

    case 0x01a0:

    case 0x01a8:

    case 0x01b0:

    case 0x01b8:

    case 0x01c0:

    case 0x01c8:

    case 0x01d0:

    case 0x01d8:

    case 0x01e0:

    case 0x01e8:

    case 0x01f0:

    case 0x01f8:

        {

            int entry = (addr - 0x0100) >> 5;

            int idx = (addr & 0x1f) >> 3;

            s->dma_regs[entry][idx] = val;

        }

        break;

    /* Memory refresh rate */

    case 0x0210:

        s->memory_refresh_rate = val;

        break;

    /* Interval timer reload */

    case 0x0228:

        s->itr = val;

        qemu_irq_lower(s->timer_irq);

        set_next_tick(s);

        break;

    /* EISA interrupt */

    case 0x0238:

        break;

    default:

        qemu_log_mask(LOG_GUEST_ERROR,

                      "rc4030: invalid write of 0x%02x at 0x%x",

                      val, (int)addr);

        break;

    }

}
