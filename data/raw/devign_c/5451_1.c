static inline void hcscale_fast_c(SwsContext *c, int16_t *dst1, int16_t *dst2,

                                  int dstWidth, const uint8_t *src1,

                                  const uint8_t *src2, int srcW, int xInc)

{

    int i;

    unsigned int xpos=0;

    for (i=0;i<dstWidth;i++) {

        register unsigned int xx=xpos>>16;

        register unsigned int xalpha=(xpos&0xFFFF)>>9;

        dst1[i]=(src1[xx]*(xalpha^127)+src1[xx+1]*xalpha);

        dst2[i]=(src2[xx]*(xalpha^127)+src2[xx+1]*xalpha);

        xpos+=xInc;





