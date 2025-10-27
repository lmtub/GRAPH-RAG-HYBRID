uint64_t HELPER(lpq)(CPUS390XState *env, uint64_t addr)

{

    uintptr_t ra = GETPC();

    uint64_t hi, lo;



    if (parallel_cpus) {

#ifndef CONFIG_ATOMIC128

        cpu_loop_exit_atomic(ENV_GET_CPU(env), ra);

#else

        int mem_idx = cpu_mmu_index(env, false);

        TCGMemOpIdx oi = make_memop_idx(MO_TEQ | MO_ALIGN_16, mem_idx);

        Int128 v = helper_atomic_ldo_be_mmu(env, addr, oi, ra);

        hi = int128_gethi(v);

        lo = int128_getlo(v);

#endif

    } else {

        check_alignment(env, addr, 16, ra);



        hi = cpu_ldq_data_ra(env, addr + 0, ra);

        lo = cpu_ldq_data_ra(env, addr + 8, ra);

    }



    env->retxl = lo;

    return hi;

}
