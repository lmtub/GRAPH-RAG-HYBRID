static unsigned long copy_strings(int argc,char ** argv,unsigned long *page,

                unsigned long p)

{

    char *tmp, *tmp1, *pag = NULL;

    int len, offset = 0;



    if (!p) {

	return 0;       /* bullet-proofing */

    }

    while (argc-- > 0) {

	if (!(tmp1 = tmp = get_user(argv+argc))) {

	    fprintf(stderr, "VFS: argc is wrong");

	    exit(-1);

	}

	while (get_user(tmp++));

	len = tmp - tmp1;

	if (p < len) {  /* this shouldn't happen - 128kB */

		return 0;

	}

	while (len) {

	    --p; --tmp; --len;

	    if (--offset < 0) {

		offset = p % TARGET_PAGE_SIZE;

                pag = (char *) page[p/TARGET_PAGE_SIZE];

                if (!pag) {

                    pag = (char *)get_free_page();

                    page[p/TARGET_PAGE_SIZE] = (unsigned long)pag;

                    if (!pag)

                        return 0;

		}

	    }

	    if (len == 0 || offset == 0) {

	        *(pag + offset) = get_user(tmp);

	    }

	    else {

	      int bytes_to_copy = (len > offset) ? offset : len;

	      tmp -= bytes_to_copy;

	      p -= bytes_to_copy;

	      offset -= bytes_to_copy;

	      len -= bytes_to_copy;

	      memcpy_fromfs(pag + offset, tmp, bytes_to_copy + 1);

	    }

	}

    }

    return p;

}
