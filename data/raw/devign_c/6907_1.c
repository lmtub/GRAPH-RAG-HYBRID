static inline PageDesc *page_find_alloc(target_ulong index)

{

    PageDesc **lp, *p;



#if TARGET_LONG_BITS > 32

    /* Host memory outside guest VM.  For 32-bit targets we have already

       excluded high addresses.  */

    if (index > ((target_ulong)L2_SIZE * L1_SIZE))

        return NULL;

#endif

    lp = &l1_map[index >> L2_BITS];

    p = *lp;

    if (!p) {

        /* allocate if not found */

#if defined(CONFIG_USER_ONLY)

        unsigned long addr;

        size_t len = sizeof(PageDesc) * L2_SIZE;

        /* Don't use qemu_malloc because it may recurse.  */

        p = mmap(0, len, PROT_READ | PROT_WRITE,

                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

        *lp = p;

        addr = h2g(p);

        if (addr == (target_ulong)addr) {

            page_set_flags(addr & TARGET_PAGE_MASK,

                           TARGET_PAGE_ALIGN(addr + len),

                           PAGE_RESERVED); 

        }

#else

        p = qemu_mallocz(sizeof(PageDesc) * L2_SIZE);

        *lp = p;

#endif

    }

    return p + (index & (L2_SIZE - 1));

}
