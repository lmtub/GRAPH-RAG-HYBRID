static inline void RENAME(bgr24ToUV)(uint8_t *dstU, uint8_t *dstV, const uint8_t *src1, const uint8_t *src2, int width, uint32_t *unused)

{

#if COMPILE_TEMPLATE_MMX

    RENAME(bgr24ToUV_mmx)(dstU, dstV, src1, width, PIX_FMT_BGR24);

#else

    int i;

    for (i=0; i<width; i++) {

        int b= src1[3*i + 0];

        int g= src1[3*i + 1];

        int r= src1[3*i + 2];



        dstU[i]= (RU*r + GU*g + BU*b + (257<<(RGB2YUV_SHIFT-1)))>>RGB2YUV_SHIFT;

        dstV[i]= (RV*r + GV*g + BV*b + (257<<(RGB2YUV_SHIFT-1)))>>RGB2YUV_SHIFT;

    }

#endif /* COMPILE_TEMPLATE_MMX */

    assert(src1 == src2);

}
