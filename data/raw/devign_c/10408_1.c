static inline void RENAME(yv12toyuy2)(const uint8_t *ysrc, const uint8_t *usrc, const uint8_t *vsrc, uint8_t *dst,

	unsigned int width, unsigned int height,

	int lumStride, int chromStride, int dstStride)

{

	//FIXME interpolate chroma

	RENAME(yuvPlanartoyuy2)(ysrc, usrc, vsrc, dst, width, height, lumStride, chromStride, dstStride, 2);

}
