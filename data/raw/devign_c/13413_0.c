static void do_dcbz(target_ulong addr, int dcache_line_size)

{

    addr &= ~(dcache_line_size - 1);

    int i;

    for (i = 0 ; i < dcache_line_size ; i += 4) {

        stl(addr + i , 0);

    }

    if (env->reserve == addr)

        env->reserve = (target_ulong)-1ULL;

}
