static inline int cris_bound_d(int v, int b)

{

	int r = v;

	asm ("bound.d\t%1, %0\n" : "+r" (r) : "ri" (b));

	return r;

}
