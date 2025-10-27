static void omap_pin_cfg_init(MemoryRegion *system_memory,

                target_phys_addr_t base,

                struct omap_mpu_state_s *mpu)

{

    memory_region_init_io(&mpu->pin_cfg_iomem, &omap_pin_cfg_ops, mpu,

                          "omap-pin-cfg", 0x800);

    memory_region_add_subregion(system_memory, base, &mpu->pin_cfg_iomem);

    omap_pin_cfg_reset(mpu);

}
