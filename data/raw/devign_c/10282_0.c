struct omap_uwire_s *omap_uwire_init(MemoryRegion *system_memory,

                target_phys_addr_t base,

                qemu_irq *irq, qemu_irq dma, omap_clk clk)

{

    struct omap_uwire_s *s = (struct omap_uwire_s *)

            g_malloc0(sizeof(struct omap_uwire_s));



    s->txirq = irq[0];

    s->rxirq = irq[1];

    s->txdrq = dma;

    omap_uwire_reset(s);



    memory_region_init_io(&s->iomem, &omap_uwire_ops, s, "omap-uwire", 0x800);

    memory_region_add_subregion(system_memory, base, &s->iomem);



    return s;

}
