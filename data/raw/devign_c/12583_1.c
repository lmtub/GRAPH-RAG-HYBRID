struct omap_gp_timer_s *omap_gp_timer_init(struct omap_target_agent_s *ta,

                qemu_irq irq, omap_clk fclk, omap_clk iclk)

{

    struct omap_gp_timer_s *s = (struct omap_gp_timer_s *)

            g_malloc0(sizeof(struct omap_gp_timer_s));



    s->ta = ta;

    s->irq = irq;

    s->clk = fclk;

    s->timer = timer_new_ns(QEMU_CLOCK_VIRTUAL, omap_gp_timer_tick, s);

    s->match = timer_new_ns(QEMU_CLOCK_VIRTUAL, omap_gp_timer_match, s);

    s->in = qemu_allocate_irq(omap_gp_timer_input, s, 0);

    omap_gp_timer_reset(s);

    omap_gp_timer_clk_setup(s);



    memory_region_init_io(&s->iomem, NULL, &omap_gp_timer_ops, s, "omap.gptimer",

                          omap_l4_region_size(ta, 0));

    omap_l4_attach(ta, 0, &s->iomem);



    return s;

}
