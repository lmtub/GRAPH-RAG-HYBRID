static uint64_t elcr_ioport_read(void *opaque, target_phys_addr_t addr,

                                 unsigned size)

{

    PICCommonState *s = opaque;

    return s->elcr;

}
