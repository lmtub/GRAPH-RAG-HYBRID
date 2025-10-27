static struct omap_sti_s *omap_sti_init(struct omap_target_agent_s *ta,

                MemoryRegion *sysmem,

                hwaddr channel_base, qemu_irq irq, omap_clk clk,

                CharDriverState *chr)

{

    struct omap_sti_s *s = (struct omap_sti_s *)

            g_malloc0(sizeof(struct omap_sti_s));



    s->irq = irq;

    omap_sti_reset(s);



    s->chr = chr ?: qemu_chr_new("null", "null", NULL);



    memory_region_init_io(&s->iomem, NULL, &omap_sti_ops, s, "omap.sti",

                          omap_l4_region_size(ta, 0));

    omap_l4_attach(ta, 0, &s->iomem);



    memory_region_init_io(&s->iomem_fifo, NULL, &omap_sti_fifo_ops, s,

                          "omap.sti.fifo", 0x10000);

    memory_region_add_subregion(sysmem, channel_base, &s->iomem_fifo);



    return s;

}
