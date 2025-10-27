static uint64_t cirrus_linear_bitblt_read(void *opaque,

                                          target_phys_addr_t addr,

                                          unsigned size)

{

    CirrusVGAState *s = opaque;

    uint32_t ret;



    /* XXX handle bitblt */

    (void)s;

    ret = 0xff;

    return ret;

}
