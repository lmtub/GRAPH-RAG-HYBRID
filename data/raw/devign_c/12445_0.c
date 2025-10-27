void omap_badwidth_write8(void *opaque, target_phys_addr_t addr,

                uint32_t value)

{

    uint8_t val8 = value;



    OMAP_8B_REG(addr);

    cpu_physical_memory_write(addr, (void *) &val8, 1);

}
