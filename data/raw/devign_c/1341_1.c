static inline void RENAME(rgb32ToY)(uint8_t *dst, uint8_t *src, int width)

{

	int i;

	for(i=0; i<width; i++)

	{

		int r=  ((uint32_t*)src)[i]&0xFF;

		int g= (((uint32_t*)src)[i]>>8)&0xFF;

		int b= (((uint32_t*)src)[i]>>16)&0xFF;



		dst[i]= ((RY*r + GY*g + BY*b + (33<<(RGB2YUV_SHIFT-1)) )>>RGB2YUV_SHIFT);

	}

}
