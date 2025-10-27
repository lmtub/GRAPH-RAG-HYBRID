void omap_badwidth_write32(void *opaque, target_phys_addr_t addr,

                uint32_t value)

{

    OMAP_32B_REG(addr);

    cpu_physical_memory_write(addr, (void *) &value, 4);

}
