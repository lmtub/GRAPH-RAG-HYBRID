static inline void RENAME(yuv422ptouyvy)(const uint8_t *ysrc, const uint8_t *usrc, const uint8_t *vsrc, uint8_t *dst,

                                         long width, long height,

                                         long lumStride, long chromStride, long dstStride)

{

    RENAME(yuvPlanartouyvy)(ysrc, usrc, vsrc, dst, width, height, lumStride, chromStride, dstStride, 1);

}
