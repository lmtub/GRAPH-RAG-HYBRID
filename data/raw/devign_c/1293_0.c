int page_check_range(target_ulong start, target_ulong len, int flags)

{

    PageDesc *p;

    target_ulong end;

    target_ulong addr;



    if (start + len < start)

        /* we've wrapped around */

        return -1;



    end = TARGET_PAGE_ALIGN(start+len); /* must do before we loose bits in the next step */

    start = start & TARGET_PAGE_MASK;



    for(addr = start; addr < end; addr += TARGET_PAGE_SIZE) {

        p = page_find(addr >> TARGET_PAGE_BITS);

        if( !p )

            return -1;

        if( !(p->flags & PAGE_VALID) )

            return -1;



        if ((flags & PAGE_READ) && !(p->flags & PAGE_READ))

            return -1;

        if (flags & PAGE_WRITE) {

            if (!(p->flags & PAGE_WRITE_ORG))

                return -1;

            /* unprotect the page if it was put read-only because it

               contains translated code */

            if (!(p->flags & PAGE_WRITE)) {

                if (!page_unprotect(addr, 0, NULL))

                    return -1;

            }

            return 0;

        }

    }

    return 0;

}
