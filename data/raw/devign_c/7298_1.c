static inline void RENAME(nv12ToUV)(uint8_t *dstU, uint8_t *dstV,

                                    const uint8_t *src1, const uint8_t *src2,

                                    int width, uint32_t *unused)

{

    RENAME(nvXXtoUV)(dstU, dstV, src1, width);

}
