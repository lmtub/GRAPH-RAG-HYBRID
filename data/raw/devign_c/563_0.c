static void cirrus_linear_bitblt_write(void *opaque,

                                       target_phys_addr_t addr,

                                       uint64_t val,

                                       unsigned size)

{

    CirrusVGAState *s = opaque;



    if (s->cirrus_srcptr != s->cirrus_srcptr_end) {

	/* bitblt */

	*s->cirrus_srcptr++ = (uint8_t) val;

	if (s->cirrus_srcptr >= s->cirrus_srcptr_end) {

	    cirrus_bitblt_cputovideo_next(s);

	}

    }

}
