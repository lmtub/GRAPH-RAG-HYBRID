static test_speed(int step)

{

    const struct pix_func* pix = pix_func;

    const int linesize = 720;

    char empty[32768];

    char* bu =(char*)(((long)empty + 32) & ~0xf);



    int sum = 0;



    while (pix->name)

    {

	int i;

        uint64_t te, ts;

        op_pixels_func func = pix->func;

	char* im = bu;



	if (!(pix->mm_flags & mm_flags))

            continue;



	printf("%30s... ", pix->name);

        fflush(stdout);

	ts = rdtsc();

	for(i=0; i<100000; i++){

	    func(im, im + 1000, linesize, 16);

	    im += step;

	    if (im > bu + 20000)

		im = bu;

	}

	te = rdtsc();

        emms();

	printf("% 9d\n", (int)(te - ts));

        sum += (te - ts) / 100000;

	if (pix->mm_flags & PAD)

            puts("");

        pix++;

    }



    printf("Total sum: %d\n", sum);

}
