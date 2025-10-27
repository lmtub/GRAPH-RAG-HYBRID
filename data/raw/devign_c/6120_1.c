static struct omap_prcm_s *omap_prcm_init(struct omap_target_agent_s *ta,

                qemu_irq mpu_int, qemu_irq dsp_int, qemu_irq iva_int,

                struct omap_mpu_state_s *mpu)

{

    struct omap_prcm_s *s = (struct omap_prcm_s *)

            g_malloc0(sizeof(struct omap_prcm_s));



    s->irq[0] = mpu_int;

    s->irq[1] = dsp_int;

    s->irq[2] = iva_int;

    s->mpu = mpu;

    omap_prcm_coldreset(s);



    memory_region_init_io(&s->iomem0, NULL, &omap_prcm_ops, s, "omap.pcrm0",

                          omap_l4_region_size(ta, 0));

    memory_region_init_io(&s->iomem1, NULL, &omap_prcm_ops, s, "omap.pcrm1",

                          omap_l4_region_size(ta, 1));

    omap_l4_attach(ta, 0, &s->iomem0);

    omap_l4_attach(ta, 1, &s->iomem1);



    return s;

}
