int omap_validate_tipb_mpui_addr(struct omap_mpu_state_s *s,

                target_phys_addr_t addr)

{

    return addr >= 0xe1010000 && addr < 0xe1020004;

}
