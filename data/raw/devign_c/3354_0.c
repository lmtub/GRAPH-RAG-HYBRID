static inline int vec_reg_offset(int regno, int element, TCGMemOp size)

{

    int offs = offsetof(CPUARMState, vfp.regs[regno * 2]);

#ifdef HOST_WORDS_BIGENDIAN

    /* This is complicated slightly because vfp.regs[2n] is

     * still the low half and  vfp.regs[2n+1] the high half

     * of the 128 bit vector, even on big endian systems.

     * Calculate the offset assuming a fully bigendian 128 bits,

     * then XOR to account for the order of the two 64 bit halves.

     */

    offs += (16 - ((element + 1) * (1 << size)));

    offs ^= 8;

#else

    offs += element * (1 << size);

#endif

    return offs;

}
