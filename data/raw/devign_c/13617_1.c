int page_check_range(target_ulong start, target_ulong len, int flags)

{

    PageDesc *p;

    target_ulong end;

    target_ulong addr;



    /* This function should never be called with addresses outside the

       guest address space.  If this assert fires, it probably indicates

       a missing call to h2g_valid.  */

#if TARGET_ABI_BITS > L1_MAP_ADDR_SPACE_BITS

    assert(start < ((abi_ulong)1 << L1_MAP_ADDR_SPACE_BITS));

#endif



    if (len == 0) {

        return 0;

    }

    if (start + len - 1 < start) {

        /* We've wrapped around.  */

        return -1;

    }



    /* must do before we loose bits in the next step */

    end = TARGET_PAGE_ALIGN(start + len);

    start = start & TARGET_PAGE_MASK;



    for (addr = start, len = end - start;

         len != 0;

         len -= TARGET_PAGE_SIZE, addr += TARGET_PAGE_SIZE) {

        p = page_find(addr >> TARGET_PAGE_BITS);

        if (!p) {

            return -1;

        }

        if (!(p->flags & PAGE_VALID)) {

            return -1;

        }



        if ((flags & PAGE_READ) && !(p->flags & PAGE_READ)) {

            return -1;

        }

        if (flags & PAGE_WRITE) {

            if (!(p->flags & PAGE_WRITE_ORG)) {

                return -1;

            }

            /* unprotect the page if it was put read-only because it

               contains translated code */

            if (!(p->flags & PAGE_WRITE)) {

                if (!page_unprotect(addr, 0, NULL)) {

                    return -1;

                }

            }

            return 0;

        }

    }

    return 0;

}
