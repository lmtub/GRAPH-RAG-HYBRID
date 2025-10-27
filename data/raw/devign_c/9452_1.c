static struct omap_mcbsp_s *omap_mcbsp_init(MemoryRegion *system_memory,

                                            hwaddr base,

                                            qemu_irq txirq, qemu_irq rxirq,

                                            qemu_irq *dma, omap_clk clk)

{

    struct omap_mcbsp_s *s = (struct omap_mcbsp_s *)

            g_malloc0(sizeof(struct omap_mcbsp_s));



    s->txirq = txirq;

    s->rxirq = rxirq;

    s->txdrq = dma[0];

    s->rxdrq = dma[1];

    s->sink_timer = timer_new_ns(QEMU_CLOCK_VIRTUAL, omap_mcbsp_sink_tick, s);

    s->source_timer = timer_new_ns(QEMU_CLOCK_VIRTUAL, omap_mcbsp_source_tick, s);

    omap_mcbsp_reset(s);



    memory_region_init_io(&s->iomem, NULL, &omap_mcbsp_ops, s, "omap-mcbsp", 0x800);

    memory_region_add_subregion(system_memory, base, &s->iomem);



    return s;

}
