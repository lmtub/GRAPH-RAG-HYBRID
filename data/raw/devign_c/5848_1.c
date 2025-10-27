static inline void tb_alloc_page(TranslationBlock *tb,

                                 unsigned int n, tb_page_addr_t page_addr)

{

    PageDesc *p;

#ifndef CONFIG_USER_ONLY

    bool page_already_protected;

#endif



    assert_memory_lock();



    tb->page_addr[n] = page_addr;

    p = page_find_alloc(page_addr >> TARGET_PAGE_BITS, 1);

    tb->page_next[n] = p->first_tb;

#ifndef CONFIG_USER_ONLY

    page_already_protected = p->first_tb != NULL;

#endif

    p->first_tb = (TranslationBlock *)((uintptr_t)tb | n);

    invalidate_page_bitmap(p);



#if defined(CONFIG_USER_ONLY)

    if (p->flags & PAGE_WRITE) {

        target_ulong addr;

        PageDesc *p2;

        int prot;



        /* force the host page as non writable (writes will have a

           page fault + mprotect overhead) */

        page_addr &= qemu_host_page_mask;

        prot = 0;

        for (addr = page_addr; addr < page_addr + qemu_host_page_size;

            addr += TARGET_PAGE_SIZE) {



            p2 = page_find(addr >> TARGET_PAGE_BITS);

            if (!p2) {

                continue;

            }

            prot |= p2->flags;

            p2->flags &= ~PAGE_WRITE;

          }

        mprotect(g2h(page_addr), qemu_host_page_size,

                 (prot & PAGE_BITS) & ~PAGE_WRITE);

#ifdef DEBUG_TB_INVALIDATE

        printf("protecting code page: 0x" TARGET_FMT_lx "\n",

               page_addr);

#endif

    }

#else

    /* if some code is already present, then the pages are already

       protected. So we handle the case where only the first TB is

       allocated in a physical page */

    if (!page_already_protected) {

        tlb_protect_code(page_addr);

    }

#endif

}
