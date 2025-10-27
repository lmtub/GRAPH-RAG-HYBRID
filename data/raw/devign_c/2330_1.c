static inline int cris_addc_m(int a, const int *b)

{

	asm volatile ("addc [%1], %0\n" : "+r" (a) : "r" (b));

	return a;

}
