MemTxResult gicv3_redist_read(void *opaque, hwaddr offset, uint64_t *data,
                              unsigned size, MemTxAttrs attrs)
{
    GICv3State *s = opaque;
    GICv3CPUState *cs;
    MemTxResult r;
    int cpuidx;
    /* This region covers all the redistributor pages; there are
     * (for GICv3) two 64K pages per CPU. At the moment they are
     * all contiguous (ie in this one region), though we might later
     * want to allow splitting of redistributor pages into several
     * blocks so we can support more CPUs.
     */
    cpuidx = offset / 0x20000;
    offset %= 0x20000;
    assert(cpuidx < s->num_cpu);
    cs = &s->cpu[cpuidx];
    switch (size) {
    case 1:
        r = gicr_readb(cs, offset, data, attrs);
        break;
    case 4:
        r = gicr_readl(cs, offset, data, attrs);
        break;
    case 8:
        r = gicr_readll(cs, offset, data, attrs);
        break;
    default:
        r = MEMTX_ERROR;
        break;
    }
    if (r == MEMTX_ERROR) {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "%s: invalid guest read at offset " TARGET_FMT_plx
                      "size %u\n", __func__, offset, size);
        trace_gicv3_redist_badread(gicv3_redist_affid(cs), offset,
                                   size, attrs.secure);
    } else {
        trace_gicv3_redist_read(gicv3_redist_affid(cs), offset, *data,
                                size, attrs.secure);
    }
    return r;
}