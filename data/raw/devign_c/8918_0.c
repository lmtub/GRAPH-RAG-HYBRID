struct omap_lcd_panel_s *omap_lcdc_init(MemoryRegion *sysmem,

                                        hwaddr base,

                                        qemu_irq irq,

                                        struct omap_dma_lcd_channel_s *dma,

                                        omap_clk clk)

{

    struct omap_lcd_panel_s *s = (struct omap_lcd_panel_s *)

            g_malloc0(sizeof(struct omap_lcd_panel_s));



    s->irq = irq;

    s->dma = dma;

    s->sysmem = sysmem;

    omap_lcdc_reset(s);



    memory_region_init_io(&s->iomem, &omap_lcdc_ops, s, "omap.lcdc", 0x100);

    memory_region_add_subregion(sysmem, base, &s->iomem);



    s->con = graphic_console_init(omap_update_display,

                                  omap_invalidate_display,

                                  omap_screen_dump, NULL, s);



    return s;

}
