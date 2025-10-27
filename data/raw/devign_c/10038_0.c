static void arm_cache_flush(abi_ulong start, abi_ulong last)

{

    abi_ulong addr, last1;



    if (last < start)

        return;

    addr = start;

    for(;;) {

        last1 = ((addr + TARGET_PAGE_SIZE) & TARGET_PAGE_MASK) - 1;

        if (last1 > last)

            last1 = last;

        tb_invalidate_page_range(addr, last1 + 1);

        if (last1 == last)

            break;

        addr = last1 + 1;

    }

}
