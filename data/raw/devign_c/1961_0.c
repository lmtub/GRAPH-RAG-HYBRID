static inline void RENAME(yuy2ToY)(uint8_t *dst, uint8_t *src, int width)

{

#ifdef HAVE_MMXFIXME

#else

	int i;

	for(i=0; i<width; i++)

		dst[i]= src[2*i];

#endif

}
