void rgb15tobgr24(const uint8_t *src, uint8_t *dst, long src_size)

{

	const uint16_t *end;

	uint8_t *d = (uint8_t *)dst;

	const uint16_t *s = (uint16_t *)src;

	end = s + src_size/2;

	while(s < end)

	{

		register uint16_t bgr;

		bgr = *s++;

		*d++ = (bgr&0x7C00)>>7;

		*d++ = (bgr&0x3E0)>>2;

		*d++ = (bgr&0x1F)<<3;

	}

}
