static void omap_gp_timer_clk_setup(struct omap_gp_timer_s *timer)

{

    omap_clk_adduser(timer->clk,

                    qemu_allocate_irqs(omap_gp_timer_clk_update, timer, 1)[0]);

    timer->rate = omap_clk_getrate(timer->clk);

}
