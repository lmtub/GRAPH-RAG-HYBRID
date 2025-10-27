static inline int popcountl(unsigned long l)

{

    return BITS_PER_LONG == 32 ? ctpop32(l) : ctpop64(l);

}
