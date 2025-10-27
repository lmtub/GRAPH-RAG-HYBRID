void palette8tobgr15(const uint8_t *src, uint8_t *dst, long num_pixels, const uint8_t *palette)

{

	long i;

	for(i=0; i<num_pixels; i++)

		((uint16_t *)dst)[i] = bswap_16(((uint16_t *)palette)[ src[i] ]);

}
