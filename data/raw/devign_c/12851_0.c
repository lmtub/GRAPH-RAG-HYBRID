static void padzero(unsigned long elf_bss)

{

        unsigned long nbyte;

        char * fpnt;



        nbyte = elf_bss & (host_page_size-1);	/* was TARGET_PAGE_SIZE - JRP */

        if (nbyte) {

	    nbyte = host_page_size - nbyte;

	    fpnt = (char *) elf_bss;

	    do {

		*fpnt++ = 0;

	    } while (--nbyte);

        }

}
