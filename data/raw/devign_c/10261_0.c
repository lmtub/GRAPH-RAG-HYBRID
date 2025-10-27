int omap_validate_emiff_addr(struct omap_mpu_state_s *s,

                target_phys_addr_t addr)

{

    return addr >= OMAP_EMIFF_BASE && addr < OMAP_EMIFF_BASE + s->sdram_size;

}
