static inline void RENAME(bgr16ToUV)(uint8_t *dstU, uint8_t *dstV, uint8_t *src1, uint8_t *src2, int width)

{

	int i;

        assert(src1==src2);

	for(i=0; i<width; i++)

	{

		int d0= ((uint32_t*)src1)[i];



		int dl= (d0&0x07E0F81F);

		int dh= ((d0>>5)&0x07C0F83F);



		int dh2= (dh>>11) + (dh<<21);

		int d= dh2 + dl;



		int b= d&0x7F;

		int r= (d>>11)&0x7F;

		int g= d>>21;

		dstU[i]= ((2*RU*r + GU*g + 2*BU*b)>>(RGB2YUV_SHIFT+1-2)) + 128;

		dstV[i]= ((2*RV*r + GV*g + 2*BV*b)>>(RGB2YUV_SHIFT+1-2)) + 128;

	}

}
