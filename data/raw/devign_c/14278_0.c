static always_inline int isnormal (float64 d)

{

    CPU_DoubleU u;



    u.d = d;



    uint32_t exp = (u.ll >> 52) & 0x7FF;

    return ((0 < exp) && (exp < 0x7FF));

}
