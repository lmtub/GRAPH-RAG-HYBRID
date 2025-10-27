static struct omap_mpu_timer_s *omap_mpu_timer_init(MemoryRegion *system_memory,

                hwaddr base,

                qemu_irq irq, omap_clk clk)

{

    struct omap_mpu_timer_s *s = (struct omap_mpu_timer_s *)

            g_malloc0(sizeof(struct omap_mpu_timer_s));



    s->irq = irq;

    s->clk = clk;

    s->timer = timer_new_ns(QEMU_CLOCK_VIRTUAL, omap_timer_tick, s);

    s->tick = qemu_bh_new(omap_timer_fire, s);

    omap_mpu_timer_reset(s);

    omap_timer_clk_setup(s);



    memory_region_init_io(&s->iomem, NULL, &omap_mpu_timer_ops, s,

                          "omap-mpu-timer", 0x100);



    memory_region_add_subregion(system_memory, base, &s->iomem);



    return s;

}
