int omap_validate_emifs_addr(struct omap_mpu_state_s *s,

                target_phys_addr_t addr)

{

    return addr >= OMAP_EMIFS_BASE && addr < OMAP_EMIFF_BASE;

}
