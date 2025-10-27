static inline unsigned long align_sigframe(unsigned long sp)

{

    unsigned long i;

    i = sp & ~3UL;

    return i;

}
