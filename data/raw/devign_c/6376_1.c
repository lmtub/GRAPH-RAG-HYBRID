static inline int cris_swap(const int mode, int x)

{

	switch (mode)

	{

		case N: asm ("swapn\t%0\n" : "+r" (x) : "0" (x)); break;

		case W: asm ("swapw\t%0\n" : "+r" (x) : "0" (x)); break;

		case B: asm ("swapb\t%0\n" : "+r" (x) : "0" (x)); break;

		case R: asm ("swapr\t%0\n" : "+r" (x) : "0" (x)); break;

		case B|R: asm ("swapbr\t%0\n" : "+r" (x) : "0" (x)); break;

		case W|R: asm ("swapwr\t%0\n" : "+r" (x) : "0" (x)); break;

		case W|B: asm ("swapwb\t%0\n" : "+r" (x) : "0" (x)); break;

		case W|B|R: asm ("swapwbr\t%0\n" : "+r" (x) : "0" (x)); break;

		case N|R: asm ("swapnr\t%0\n" : "+r" (x) : "0" (x)); break;

		case N|B: asm ("swapnb\t%0\n" : "+r" (x) : "0" (x)); break;

		case N|B|R: asm ("swapnbr\t%0\n" : "+r" (x) : "0" (x)); break;

		case N|W: asm ("swapnw\t%0\n" : "+r" (x) : "0" (x)); break;

		default:

			err();

			break;

	}

	return x;

}
