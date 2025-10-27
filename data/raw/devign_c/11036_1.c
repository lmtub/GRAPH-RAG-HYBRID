static inline int *DEC_UPAIR(int *dst, unsigned idx, unsigned sign)

{

    dst[0] = (idx & 15) * (1 - (sign & 0xFFFFFFFE));

    dst[1] = (idx >> 4 & 15) * (1 - ((sign & 1) << 1));



    return dst + 2;

}
