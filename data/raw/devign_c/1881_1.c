static struct omap_sysctl_s *omap_sysctl_init(struct omap_target_agent_s *ta,

                omap_clk iclk, struct omap_mpu_state_s *mpu)

{

    struct omap_sysctl_s *s = (struct omap_sysctl_s *)

            g_malloc0(sizeof(struct omap_sysctl_s));



    s->mpu = mpu;

    omap_sysctl_reset(s);



    memory_region_init_io(&s->iomem, NULL, &omap_sysctl_ops, s, "omap.sysctl",

                          omap_l4_region_size(ta, 0));

    omap_l4_attach(ta, 0, &s->iomem);



    return s;

}
