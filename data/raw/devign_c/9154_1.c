static struct omap_lpg_s *omap_lpg_init(MemoryRegion *system_memory,

                                        hwaddr base, omap_clk clk)

{

    struct omap_lpg_s *s = (struct omap_lpg_s *)

            g_malloc0(sizeof(struct omap_lpg_s));



    s->tm = timer_new_ms(QEMU_CLOCK_VIRTUAL, omap_lpg_tick, s);



    omap_lpg_reset(s);



    memory_region_init_io(&s->iomem, NULL, &omap_lpg_ops, s, "omap-lpg", 0x800);

    memory_region_add_subregion(system_memory, base, &s->iomem);



    omap_clk_adduser(clk, qemu_allocate_irqs(omap_lpg_clk_update, s, 1)[0]);



    return s;

}
