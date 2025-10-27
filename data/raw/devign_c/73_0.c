static void omap_pwl_init(target_phys_addr_t base, struct omap_mpu_state_s *s,

                omap_clk clk)

{

    int iomemtype;



    s->pwl.base = base;

    omap_pwl_reset(s);



    iomemtype = cpu_register_io_memory(0, omap_pwl_readfn,

                    omap_pwl_writefn, s);

    cpu_register_physical_memory(s->pwl.base, 0x800, iomemtype);



    omap_clk_adduser(clk, qemu_allocate_irqs(omap_pwl_clk_update, s, 1)[0]);

}
