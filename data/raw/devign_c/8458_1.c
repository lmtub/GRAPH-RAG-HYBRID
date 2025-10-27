struct omap_dss_s *omap_dss_init(struct omap_target_agent_s *ta,

                MemoryRegion *sysmem,

                hwaddr l3_base,

                qemu_irq irq, qemu_irq drq,

                omap_clk fck1, omap_clk fck2, omap_clk ck54m,

                omap_clk ick1, omap_clk ick2)

{

    struct omap_dss_s *s = (struct omap_dss_s *)

            g_malloc0(sizeof(struct omap_dss_s));



    s->irq = irq;

    s->drq = drq;

    omap_dss_reset(s);



    memory_region_init_io(&s->iomem_diss1, NULL, &omap_diss_ops, s, "omap.diss1",

                          omap_l4_region_size(ta, 0));

    memory_region_init_io(&s->iomem_disc1, NULL, &omap_disc_ops, s, "omap.disc1",

                          omap_l4_region_size(ta, 1));

    memory_region_init_io(&s->iomem_rfbi1, NULL, &omap_rfbi_ops, s, "omap.rfbi1",

                          omap_l4_region_size(ta, 2));

    memory_region_init_io(&s->iomem_venc1, NULL, &omap_venc_ops, s, "omap.venc1",

                          omap_l4_region_size(ta, 3));

    memory_region_init_io(&s->iomem_im3, NULL, &omap_im3_ops, s,

                          "omap.im3", 0x1000);



    omap_l4_attach(ta, 0, &s->iomem_diss1);

    omap_l4_attach(ta, 1, &s->iomem_disc1);

    omap_l4_attach(ta, 2, &s->iomem_rfbi1);

    omap_l4_attach(ta, 3, &s->iomem_venc1);

    memory_region_add_subregion(sysmem, l3_base, &s->iomem_im3);



#if 0

    s->state = graphic_console_init(omap_update_display,

                                    omap_invalidate_display, omap_screen_dump, s);

#endif



    return s;

}
