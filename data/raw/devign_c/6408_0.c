void HELPER(stpq)(CPUS390XState *env, uint64_t addr,

                  uint64_t low, uint64_t high)

{

    uintptr_t ra = GETPC();



    if (parallel_cpus) {

#ifndef CONFIG_ATOMIC128

        cpu_loop_exit_atomic(ENV_GET_CPU(env), ra);

#else

        int mem_idx = cpu_mmu_index(env, false);

        TCGMemOpIdx oi = make_memop_idx(MO_TEQ | MO_ALIGN_16, mem_idx);



        Int128 v = int128_make128(low, high);

        helper_atomic_sto_be_mmu(env, addr, v, oi, ra);

#endif

    } else {

        check_alignment(env, addr, 16, ra);



        cpu_stq_data_ra(env, addr + 0, high, ra);

        cpu_stq_data_ra(env, addr + 8, low, ra);

    }

}
