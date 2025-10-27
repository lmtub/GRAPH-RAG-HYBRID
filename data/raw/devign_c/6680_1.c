static inline int cris_addc(int a, const int b)

{

	asm ("addc\t%1, %0\n" : "+r" (a) : "r" (b));

	return a;

}
