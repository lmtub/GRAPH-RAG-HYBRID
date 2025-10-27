static inline int cris_bound_b(int v, int b)

{

	int r = v;

	asm ("bound.b\t%1, %0\n" : "+r" (r) : "ri" (b));

	return r;

}
