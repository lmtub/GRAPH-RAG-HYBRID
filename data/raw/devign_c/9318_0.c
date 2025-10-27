uint32_t omap_badwidth_read16(void *opaque, target_phys_addr_t addr)

{

    uint16_t ret;



    OMAP_16B_REG(addr);

    cpu_physical_memory_read(addr, (void *) &ret, 2);

    return ret;

}
