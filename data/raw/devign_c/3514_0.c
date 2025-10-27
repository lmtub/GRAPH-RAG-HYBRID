uint32_t omap_badwidth_read32(void *opaque, target_phys_addr_t addr)

{

    OMAP_32B_REG(addr);

    return 0;

}
