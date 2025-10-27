void rgb8tobgr8(const uint8_t *src, uint8_t *dst, long src_size)

{

	long i;

	long num_pixels = src_size;

	for(i=0; i<num_pixels; i++)

	{

	    unsigned b,g,r;

	    register uint8_t rgb;

	    rgb = src[i];

	    r = (rgb&0x07);

	    g = (rgb&0x38)>>3;

	    b = (rgb&0xC0)>>6;

	    dst[i] = ((b<<1)&0x07) | ((g&0x07)<<3) | ((r&0x03)<<6);

	}

}
