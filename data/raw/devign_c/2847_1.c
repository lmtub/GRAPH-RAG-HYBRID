static inline void RENAME(yuv2nv12X)(SwsContext *c, int16_t *lumFilter, int16_t **lumSrc, int lumFilterSize,

				     int16_t *chrFilter, int16_t **chrSrc, int chrFilterSize,

				     uint8_t *dest, uint8_t *uDest, int dstW, int chrDstW, int dstFormat)

{

yuv2nv12XinC(lumFilter, lumSrc, lumFilterSize,

	     chrFilter, chrSrc, chrFilterSize,

	     dest, uDest, dstW, chrDstW, dstFormat);

}
