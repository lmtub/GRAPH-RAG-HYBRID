uint32_t omap_badwidth_read8(void *opaque, target_phys_addr_t addr)

{

    uint8_t ret;



    OMAP_8B_REG(addr);

    cpu_physical_memory_read(addr, (void *) &ret, 1);

    return ret;

}
