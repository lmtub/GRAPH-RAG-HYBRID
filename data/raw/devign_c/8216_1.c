struct omap_gpmc_s *omap_gpmc_init(target_phys_addr_t base, qemu_irq irq)

{

    struct omap_gpmc_s *s = (struct omap_gpmc_s *)

            g_malloc0(sizeof(struct omap_gpmc_s));



    memory_region_init_io(&s->iomem, &omap_gpmc_ops, s, "omap-gpmc", 0x1000);

    memory_region_add_subregion(get_system_memory(), base, &s->iomem);




    omap_gpmc_reset(s);



    return s;

}