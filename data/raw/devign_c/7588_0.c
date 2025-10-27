static struct omap_lpg_s *omap_lpg_init(target_phys_addr_t base, omap_clk clk)

{

    int iomemtype;

    struct omap_lpg_s *s = (struct omap_lpg_s *)

            qemu_mallocz(sizeof(struct omap_lpg_s));



    s->tm = qemu_new_timer(rt_clock, omap_lpg_tick, s);



    omap_lpg_reset(s);



    iomemtype = cpu_register_io_memory(omap_lpg_readfn,

                    omap_lpg_writefn, s, DEVICE_NATIVE_ENDIAN);

    cpu_register_physical_memory(base, 0x800, iomemtype);



    omap_clk_adduser(clk, qemu_allocate_irqs(omap_lpg_clk_update, s, 1)[0]);



    return s;

}
