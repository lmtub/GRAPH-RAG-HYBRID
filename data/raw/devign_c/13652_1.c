QGuestAllocator *pc_alloc_init(void)
{
    PCAlloc *s = g_malloc0(sizeof(*s));
    uint64_t ram_size;
    QFWCFG *fw_cfg = pc_fw_cfg_init();
    s->alloc.alloc = pc_alloc;
    s->alloc.free = pc_free;
    ram_size = qfw_cfg_get_u64(fw_cfg, FW_CFG_RAM_SIZE);
    /* Start at 1MB */
    s->start = 1 << 20;
    /* Respect PCI hole */
    s->end = MIN(ram_size, 0xE0000000);
    return &s->alloc;
}