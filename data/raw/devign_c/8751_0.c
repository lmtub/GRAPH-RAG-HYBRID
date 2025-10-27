void stw_be_phys(target_phys_addr_t addr, uint32_t val)

{

    stw_phys_internal(addr, val, DEVICE_BIG_ENDIAN);

}
