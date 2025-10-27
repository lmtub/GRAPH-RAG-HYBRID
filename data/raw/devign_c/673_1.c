static struct omap_tipb_bridge_s *omap_tipb_bridge_init(

    MemoryRegion *memory, hwaddr base,

    qemu_irq abort_irq, omap_clk clk)

{

    struct omap_tipb_bridge_s *s = (struct omap_tipb_bridge_s *)

            g_malloc0(sizeof(struct omap_tipb_bridge_s));



    s->abort = abort_irq;

    omap_tipb_bridge_reset(s);



    memory_region_init_io(&s->iomem, NULL, &omap_tipb_bridge_ops, s,

                          "omap-tipb-bridge", 0x100);

    memory_region_add_subregion(memory, base, &s->iomem);



    return s;

}
