static inline void RENAME(bgr32ToUV)(uint8_t *dstU, uint8_t *dstV, uint8_t *src1, uint8_t *src2, int width)

{

	int i;

        assert(src1 == src2);

	for(i=0; i<width; i++)

	{

		const int a= ((uint32_t*)src1)[2*i+0];

		const int e= ((uint32_t*)src1)[2*i+1];

		const int l= (a&0xFF00FF) + (e&0xFF00FF);

		const int h= (a&0x00FF00) + (e&0x00FF00);

 		const int b=  l&0x3FF;

		const int g=  h>>8;

		const int r=  l>>16;



		dstU[i]= ((RU*r + GU*g + BU*b)>>(RGB2YUV_SHIFT+1)) + 128;

		dstV[i]= ((RV*r + GV*g + BV*b)>>(RGB2YUV_SHIFT+1)) + 128;

	}

}
