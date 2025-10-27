uint64_t HELPER(paired_cmpxchg64_le)(CPUARMState *env, uint64_t addr,

                                     uint64_t new_lo, uint64_t new_hi)

{

    uintptr_t ra = GETPC();

    Int128 oldv, cmpv, newv;

    bool success;



    cmpv = int128_make128(env->exclusive_val, env->exclusive_high);

    newv = int128_make128(new_lo, new_hi);



    if (parallel_cpus) {

#ifndef CONFIG_ATOMIC128

        cpu_loop_exit_atomic(ENV_GET_CPU(env), ra);

#else

        int mem_idx = cpu_mmu_index(env, false);

        TCGMemOpIdx oi = make_memop_idx(MO_LEQ | MO_ALIGN_16, mem_idx);

        oldv = helper_atomic_cmpxchgo_le_mmu(env, addr, cmpv, newv, oi, ra);

        success = int128_eq(oldv, cmpv);

#endif

    } else {

        uint64_t o0, o1;



#ifdef CONFIG_USER_ONLY

        /* ??? Enforce alignment.  */

        uint64_t *haddr = g2h(addr);

        o0 = ldq_le_p(haddr + 0);

        o1 = ldq_le_p(haddr + 1);

        oldv = int128_make128(o0, o1);



        success = int128_eq(oldv, cmpv);

        if (success) {

            stq_le_p(haddr + 0, int128_getlo(newv));

            stq_le_p(haddr + 1, int128_gethi(newv));

        }

#else

        int mem_idx = cpu_mmu_index(env, false);

        TCGMemOpIdx oi0 = make_memop_idx(MO_LEQ | MO_ALIGN_16, mem_idx);

        TCGMemOpIdx oi1 = make_memop_idx(MO_LEQ, mem_idx);



        o0 = helper_le_ldq_mmu(env, addr + 0, oi0, ra);

        o1 = helper_le_ldq_mmu(env, addr + 8, oi1, ra);

        oldv = int128_make128(o0, o1);



        success = int128_eq(oldv, cmpv);

        if (success) {

            helper_le_stq_mmu(env, addr + 0, int128_getlo(newv), oi1, ra);

            helper_le_stq_mmu(env, addr + 8, int128_gethi(newv), oi1, ra);

        }

#endif

    }



    return !success;

}
