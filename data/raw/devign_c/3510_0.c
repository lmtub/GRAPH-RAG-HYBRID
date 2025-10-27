void stw_le_phys(target_phys_addr_t addr, uint32_t val)

{

    stw_phys_internal(addr, val, DEVICE_LITTLE_ENDIAN);

}
