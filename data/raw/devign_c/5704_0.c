void HELPER(cdsg)(CPUS390XState *env, uint64_t addr,

                  uint32_t r1, uint32_t r3)

{

    uintptr_t ra = GETPC();

    Int128 cmpv = int128_make128(env->regs[r1 + 1], env->regs[r1]);

    Int128 newv = int128_make128(env->regs[r3 + 1], env->regs[r3]);

    Int128 oldv;

    bool fail;



    if (parallel_cpus) {

#ifndef CONFIG_ATOMIC128

        cpu_loop_exit_atomic(ENV_GET_CPU(env), ra);

#else

        int mem_idx = cpu_mmu_index(env, false);

        TCGMemOpIdx oi = make_memop_idx(MO_TEQ | MO_ALIGN_16, mem_idx);

        oldv = helper_atomic_cmpxchgo_be_mmu(env, addr, cmpv, newv, oi, ra);

        fail = !int128_eq(oldv, cmpv);

#endif

    } else {

        uint64_t oldh, oldl;



        check_alignment(env, addr, 16, ra);



        oldh = cpu_ldq_data_ra(env, addr + 0, ra);

        oldl = cpu_ldq_data_ra(env, addr + 8, ra);



        oldv = int128_make128(oldl, oldh);

        fail = !int128_eq(oldv, cmpv);

        if (fail) {

            newv = oldv;

        }



        cpu_stq_data_ra(env, addr + 0, int128_gethi(newv), ra);

        cpu_stq_data_ra(env, addr + 8, int128_getlo(newv), ra);

    }



    env->cc_op = fail;

    env->regs[r1] = int128_gethi(oldv);

    env->regs[r1 + 1] = int128_getlo(oldv);

}
