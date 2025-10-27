static void nvram_writel (void *opaque, target_phys_addr_t addr, uint32_t value)

{

    M48t59State *NVRAM = opaque;



    m48t59_write(NVRAM, addr, (value >> 24) & 0xff);

    m48t59_write(NVRAM, addr + 1, (value >> 16) & 0xff);

    m48t59_write(NVRAM, addr + 2, (value >> 8) & 0xff);

    m48t59_write(NVRAM, addr + 3, value & 0xff);

}
