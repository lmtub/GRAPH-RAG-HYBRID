static MemTxResult nvic_sysreg_write(void *opaque, hwaddr addr,

                                     uint64_t value, unsigned size,

                                     MemTxAttrs attrs)

{

    NVICState *s = (NVICState *)opaque;

    uint32_t offset = addr;

    unsigned i, startvec, end;

    unsigned setval = 0;



    trace_nvic_sysreg_write(addr, value, size);



    if (attrs.user && !nvic_user_access_ok(s, addr)) {

        /* Generate BusFault for unprivileged accesses */

        return MEMTX_ERROR;

    }



    switch (offset) {

    case 0x100 ... 0x13f: /* NVIC Set enable */

        offset += 0x80;

        setval = 1;

        /* fall through */

    case 0x180 ... 0x1bf: /* NVIC Clear enable */

        startvec = 8 * (offset - 0x180) + NVIC_FIRST_IRQ;



        for (i = 0, end = size * 8; i < end && startvec + i < s->num_irq; i++) {

            if (value & (1 << i)) {

                s->vectors[startvec + i].enabled = setval;

            }

        }

        nvic_irq_update(s);

        return MEMTX_OK;

    case 0x200 ... 0x23f: /* NVIC Set pend */

        /* the special logic in armv7m_nvic_set_pending()

         * is not needed since IRQs are never escalated

         */

        offset += 0x80;

        setval = 1;

        /* fall through */

    case 0x280 ... 0x2bf: /* NVIC Clear pend */

        startvec = 8 * (offset - 0x280) + NVIC_FIRST_IRQ; /* vector # */



        for (i = 0, end = size * 8; i < end && startvec + i < s->num_irq; i++) {

            if (value & (1 << i)) {

                s->vectors[startvec + i].pending = setval;

            }

        }

        nvic_irq_update(s);

        return MEMTX_OK;

    case 0x300 ... 0x33f: /* NVIC Active */

        return MEMTX_OK; /* R/O */

    case 0x400 ... 0x5ef: /* NVIC Priority */

        startvec = 8 * (offset - 0x400) + NVIC_FIRST_IRQ; /* vector # */



        for (i = 0; i < size && startvec + i < s->num_irq; i++) {

            set_prio(s, startvec + i, (value >> (i * 8)) & 0xff);

        }

        nvic_irq_update(s);

        return MEMTX_OK;

    case 0xd18 ... 0xd23: /* System Handler Priority.  */

        for (i = 0; i < size; i++) {

            unsigned hdlidx = (offset - 0xd14) + i;

            set_prio(s, hdlidx, (value >> (i * 8)) & 0xff);

        }

        nvic_irq_update(s);

        return MEMTX_OK;

    }

    if (size == 4) {

        nvic_writel(s, offset, value);

        return MEMTX_OK;

    }

    qemu_log_mask(LOG_GUEST_ERROR,

                  "NVIC: Bad write of size %d at offset 0x%x\n", size, offset);

    /* This is UNPREDICTABLE; treat as RAZ/WI */

    return MEMTX_OK;

}
