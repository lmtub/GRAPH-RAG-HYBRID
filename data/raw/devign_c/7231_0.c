static inline int fp_reg_offset(int regno, TCGMemOp size)

{

    int offs = offsetof(CPUARMState, vfp.regs[regno * 2]);

#ifdef HOST_WORDS_BIGENDIAN

    offs += (8 - (1 << size));

#endif

    return offs;

}
