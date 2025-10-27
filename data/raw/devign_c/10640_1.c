static inline int cris_addc_pi_m(int a, int **b)

{

	asm volatile ("addc [%1+], %0\n" : "+r" (a), "+b" (*b));

	return a;

}
