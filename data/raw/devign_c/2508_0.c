static void omap_clkm_init(MemoryRegion *memory, target_phys_addr_t mpu_base,

                target_phys_addr_t dsp_base, struct omap_mpu_state_s *s)

{

    memory_region_init_io(&s->clkm_iomem, &omap_clkm_ops, s,

                          "omap-clkm", 0x100);

    memory_region_init_io(&s->clkdsp_iomem, &omap_clkdsp_ops, s,

                          "omap-clkdsp", 0x1000);



    s->clkm.arm_idlect1 = 0x03ff;

    s->clkm.arm_idlect2 = 0x0100;

    s->clkm.dsp_idlect1 = 0x0002;

    omap_clkm_reset(s);

    s->clkm.cold_start = 0x3a;



    memory_region_add_subregion(memory, mpu_base, &s->clkm_iomem);

    memory_region_add_subregion(memory, dsp_base, &s->clkdsp_iomem);

}
