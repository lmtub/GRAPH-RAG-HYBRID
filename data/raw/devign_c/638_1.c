static abi_ulong copy_elf_strings(int argc,char ** argv, void **page,

                                  abi_ulong p)

{

    char *tmp, *tmp1, *pag = NULL;

    int len, offset = 0;



    if (!p) {

        return 0;       /* bullet-proofing */

    }

    while (argc-- > 0) {

        tmp = argv[argc];

        if (!tmp) {

            fprintf(stderr, "VFS: argc is wrong");

            exit(-1);

        }

        tmp1 = tmp;

        while (*tmp++);

        len = tmp - tmp1;

        if (p < len) {  /* this shouldn't happen - 128kB */

            return 0;

        }

        while (len) {

            --p; --tmp; --len;

            if (--offset < 0) {

                offset = p % TARGET_PAGE_SIZE;

                pag = (char *)page[p/TARGET_PAGE_SIZE];

                if (!pag) {

                    pag = (char *)malloc(TARGET_PAGE_SIZE);

                    memset(pag, 0, TARGET_PAGE_SIZE);

                    page[p/TARGET_PAGE_SIZE] = pag;

                    if (!pag)

                        return 0;

                }

            }

            if (len == 0 || offset == 0) {

                *(pag + offset) = *tmp;

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
