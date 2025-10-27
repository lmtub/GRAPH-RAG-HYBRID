void palette8torgb24(const uint8_t *src, uint8_t *dst, long num_pixels, const uint8_t *palette)

{

	long i;

/*

	writes 1 byte o much and might cause alignment issues on some architectures?

	for(i=0; i<num_pixels; i++)

		((unsigned *)(&dst[i*3])) = ((unsigned *)palette)[ src[i] ];

*/

	for(i=0; i<num_pixels; i++)

	{

		//FIXME slow?

		dst[0]= palette[ src[i]*4+2 ];

		dst[1]= palette[ src[i]*4+1 ];

		dst[2]= palette[ src[i]*4+0 ];

		dst+= 3;

	}

}
