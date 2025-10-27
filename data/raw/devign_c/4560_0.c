static uint64_t lance_mem_read(void *opaque, target_phys_addr_t addr,

                               unsigned size)

{

    SysBusPCNetState *d = opaque;

    uint32_t val;



    val = pcnet_ioport_readw(&d->state, addr);

    trace_lance_mem_readw(addr, val & 0xffff);

    return val & 0xffff;

}
