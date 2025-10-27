int mmap_frag(unsigned long host_start,

               unsigned long start, unsigned long end,

               int prot, int flags, int fd, unsigned long offset)

{

    unsigned long host_end, ret, addr;

    int prot1, prot_new;



    host_end = host_start + qemu_host_page_size;



    /* get the protection of the target pages outside the mapping */

    prot1 = 0;

    for(addr = host_start; addr < host_end; addr++) {

        if (addr < start || addr >= end)

            prot1 |= page_get_flags(addr);

    }



    if (prot1 == 0) {

        /* no page was there, so we allocate one */

        ret = (long)mmap((void *)host_start, qemu_host_page_size, prot,

                         flags | MAP_ANONYMOUS, -1, 0);

        if (ret == -1)

            return ret;

    }

    prot1 &= PAGE_BITS;



    prot_new = prot | prot1;

    if (!(flags & MAP_ANONYMOUS)) {

        /* msync() won't work here, so we return an error if write is

           possible while it is a shared mapping */

#ifndef __APPLE__

        if ((flags & MAP_TYPE) == MAP_SHARED &&

#else

        if ((flags &  MAP_SHARED) &&

#endif

            (prot & PROT_WRITE))

            return -EINVAL;



        /* adjust protection to be able to read */

        if (!(prot1 & PROT_WRITE))

            mprotect((void *)host_start, qemu_host_page_size, prot1 | PROT_WRITE);



        /* read the corresponding file data */

        pread(fd, (void *)start, end - start, offset);



        /* put final protection */

        if (prot_new != (prot1 | PROT_WRITE))

            mprotect((void *)host_start, qemu_host_page_size, prot_new);

    } else {

        /* just update the protection */

        if (prot_new != prot1) {

            mprotect((void *)host_start, qemu_host_page_size, prot_new);

        }

    }

    return 0;

}
