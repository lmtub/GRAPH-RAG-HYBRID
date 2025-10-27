static void omap_ulpd_pm_init(MemoryRegion *system_memory,

                target_phys_addr_t base,

                struct omap_mpu_state_s *mpu)

{

    memory_region_init_io(&mpu->ulpd_pm_iomem, &omap_ulpd_pm_ops, mpu,

                          "omap-ulpd-pm", 0x800);

    memory_region_add_subregion(system_memory, base, &mpu->ulpd_pm_iomem);

    omap_ulpd_pm_reset(mpu);

}
