static int omap_validate_emifs_addr(struct omap_mpu_state_s *s,

                target_phys_addr_t addr)

{

    return range_covers_byte(OMAP_EMIFS_BASE, OMAP_EMIFF_BASE - OMAP_EMIFS_BASE,

                             addr);

}
