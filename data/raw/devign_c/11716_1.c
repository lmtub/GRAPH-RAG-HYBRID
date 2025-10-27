abi_long do_brk(abi_ulong new_brk)

{

    abi_ulong brk_page;

    abi_long mapped_addr;

    int	new_alloc_size;



    if (!new_brk)

        return target_brk;

    if (new_brk < target_original_brk)

        return target_brk;



    brk_page = HOST_PAGE_ALIGN(target_brk);



    /* If the new brk is less than this, set it and we're done... */

    if (new_brk < brk_page) {

	target_brk = new_brk;

    	return target_brk;

    }



    /* We need to allocate more memory after the brk... Note that

     * we don't use MAP_FIXED because that will map over the top of

     * any existing mapping (like the one with the host libc or qemu

     * itself); instead we treat "mapped but at wrong address" as

     * a failure and unmap again.

     */

    new_alloc_size = HOST_PAGE_ALIGN(new_brk - brk_page + 1);

    mapped_addr = get_errno(target_mmap(brk_page, new_alloc_size,

                                        PROT_READ|PROT_WRITE,

                                        MAP_ANON|MAP_PRIVATE, 0, 0));



    if (mapped_addr == brk_page) {

        target_brk = new_brk;

        return target_brk;

    } else if (mapped_addr != -1) {

        /* Mapped but at wrong address, meaning there wasn't actually

         * enough space for this brk.

         */

        target_munmap(mapped_addr, new_alloc_size);

        mapped_addr = -1;

    }



#if defined(TARGET_ALPHA)

    /* We (partially) emulate OSF/1 on Alpha, which requires we

       return a proper errno, not an unchanged brk value.  */

    return -TARGET_ENOMEM;

#endif

    /* For everything else, return the previous break. */

    return target_brk;

}
