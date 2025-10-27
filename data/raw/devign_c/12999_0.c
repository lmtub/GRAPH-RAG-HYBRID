static uint32_t nvram_readw (void *opaque, target_phys_addr_t addr)

{

    M48t59State *NVRAM = opaque;

    uint32_t retval;



    retval = m48t59_read(NVRAM, addr) << 8;

    retval |= m48t59_read(NVRAM, addr + 1);

    return retval;

}
