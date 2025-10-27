static inline int cris_lz(int x)

{

	int r;

	asm ("lz\t%1, %0\n" : "=r" (r) : "r" (x));

	return r;

}
