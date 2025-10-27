static inline void RENAME(yuy2ToUV)(uint8_t *dstU, uint8_t *dstV, uint8_t *src1, uint8_t *src2, int width)

{

#ifdef HAVE_MMXFIXME

#else

	int i;

	for(i=0; i<width; i++)

	{

		dstU[i]= (src1[4*i + 1] + src2[4*i + 1])>>1;

		dstV[i]= (src1[4*i + 3] + src2[4*i + 3])>>1;

	}

#endif

}
