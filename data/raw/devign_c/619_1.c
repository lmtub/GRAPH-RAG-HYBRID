static void do_dcbz(CPUPPCState *env, target_ulong addr, int dcache_line_size,

                    uintptr_t raddr)

{

    int i;



    addr &= ~(dcache_line_size - 1);

    for (i = 0; i < dcache_line_size; i += 4) {

        cpu_stl_data_ra(env, addr + i, 0, raddr);

    }

    if (env->reserve_addr == addr) {

        env->reserve_addr = (target_ulong)-1ULL;

    }

}
