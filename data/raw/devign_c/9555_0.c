void check_file_unfixed_eof_mmaps(void)

{

	char *cp;

	unsigned int *p1;

	uintptr_t p;

	int i;



	fprintf (stderr, "%s", __func__);

	for (i = 0; i < 0x10; i++)

	{

		p1 = mmap(NULL, pagesize, PROT_READ, 

			  MAP_PRIVATE, 

			  test_fd, 

			  (test_fsize - sizeof *p1) & ~pagemask);



		fail_unless (p1 != MAP_FAILED);



		/* Make sure we get pages aligned with the pagesize. The

		   target expects this.  */

		p = (uintptr_t) p1;

		fail_unless ((p & pagemask) == 0);

		/* Verify that the file maps was made correctly.  */

		fail_unless (p1[(test_fsize & pagemask) / sizeof *p1 - 1]

			     == ((test_fsize - sizeof *p1) / sizeof *p1));



		/* Verify that the end of page is accessable and zeroed.  */

		cp = (void *) p1;

		fail_unless (cp[pagesize - 4] == 0);

		munmap (p1, pagesize);

	}

	fprintf (stderr, " passed\n");

}
