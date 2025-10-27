static inline int *DEC_UQUAD(int *dst, unsigned idx, unsigned sign)

{

    unsigned nz = idx >> 12;



    dst[0] = (idx & 3) * (1 + (((int)sign >> 31) << 1));

    sign <<= nz & 1;

    nz >>= 1;

    dst[1] = (idx >> 2 & 3) * (1 + (((int)sign >> 31) << 1));

    sign <<= nz & 1;

    nz >>= 1;

    dst[2] = (idx >> 4 & 3) * (1 + (((int)sign >> 31) << 1));

    sign <<= nz & 1;

    nz >>= 1;

    dst[3] = (idx >> 6 & 3) * (1 + (((int)sign >> 31) << 1));



    return dst + 4;

}
