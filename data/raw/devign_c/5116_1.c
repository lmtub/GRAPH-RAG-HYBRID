static uint32_t omap2_gpio_module_readp(void *opaque, target_phys_addr_t addr)

{

    return omap2_gpio_module_readp(opaque, addr) >> ((addr & 3) << 3);

}
