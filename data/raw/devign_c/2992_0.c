static int omap_validate_local_addr(struct omap_mpu_state_s *s,

                target_phys_addr_t addr)

{

    return addr >= OMAP_LOCALBUS_BASE && addr < OMAP_LOCALBUS_BASE + 0x1000000;

}
