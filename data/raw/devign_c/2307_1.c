void palette8torgb16(const uint8_t *src, uint8_t *dst, unsigned num_pixels, const uint8_t *palette)

{

	unsigned i;

	for(i=0; i<num_pixels; i++)

		((uint16_t *)dst)[i] = ((uint16_t *)palette)[ src[i] ];

}
