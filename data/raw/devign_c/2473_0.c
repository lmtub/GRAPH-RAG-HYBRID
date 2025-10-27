int omap_validate_imif_addr(struct omap_mpu_state_s *s,

                target_phys_addr_t addr)

{

    return addr >= OMAP_IMIF_BASE && addr < OMAP_IMIF_BASE + s->sram_size;

}
