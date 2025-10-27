uint32_t lduw_be_phys(target_phys_addr_t addr)

{

    return lduw_phys_internal(addr, DEVICE_BIG_ENDIAN);

}
