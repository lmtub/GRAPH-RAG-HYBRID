static void lance_mem_write(void *opaque, target_phys_addr_t addr,

                            uint64_t val, unsigned size)

{

    SysBusPCNetState *d = opaque;



    trace_lance_mem_writew(addr, val & 0xffff);

    pcnet_ioport_writew(&d->state, addr, val & 0xffff);

}
