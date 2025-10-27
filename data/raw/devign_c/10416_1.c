int page_unprotect(target_ulong address, uintptr_t pc, void *puc)

{

    unsigned int prot;

    PageDesc *p;

    target_ulong host_start, host_end, addr;



    /* Technically this isn't safe inside a signal handler.  However we

       know this only ever happens in a synchronous SEGV handler, so in

       practice it seems to be ok.  */

    mmap_lock();



    p = page_find(address >> TARGET_PAGE_BITS);

    if (!p) {

        mmap_unlock();

        return 0;

    }



    /* if the page was really writable, then we change its

       protection back to writable */

    if ((p->flags & PAGE_WRITE_ORG) && !(p->flags & PAGE_WRITE)) {

        host_start = address & qemu_host_page_mask;

        host_end = host_start + qemu_host_page_size;



        prot = 0;

        for (addr = host_start ; addr < host_end ; addr += TARGET_PAGE_SIZE) {

            p = page_find(addr >> TARGET_PAGE_BITS);

            p->flags |= PAGE_WRITE;

            prot |= p->flags;



            /* and since the content will be modified, we must invalidate

               the corresponding translated code. */

            tb_invalidate_phys_page(addr, pc, puc);

#ifdef DEBUG_TB_CHECK

            tb_invalidate_check(addr);

#endif

        }

        mprotect((void *)g2h(host_start), qemu_host_page_size,

                 prot & PAGE_BITS);



        mmap_unlock();

        return 1;

    }

    mmap_unlock();

    return 0;

}
