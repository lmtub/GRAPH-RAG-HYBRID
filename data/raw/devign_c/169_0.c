static void elcr_ioport_write(void *opaque, target_phys_addr_t addr,

                              uint64_t val, unsigned size)

{

    PICCommonState *s = opaque;

    s->elcr = val & s->elcr_mask;

}
