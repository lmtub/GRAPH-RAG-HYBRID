static inline void cris_fidx_d(unsigned int x)

{

	register unsigned int v asm("$r10") = x;

	asm ("fidxd\t[%0]\n" : : "r" (v) );

}
