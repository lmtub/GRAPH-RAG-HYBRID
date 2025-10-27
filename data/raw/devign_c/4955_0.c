uint32_t HELPER(csst)(CPUS390XState *env, uint32_t r3, uint64_t a1, uint64_t a2)

{

#if !defined(CONFIG_USER_ONLY) || defined(CONFIG_ATOMIC128)

    uint32_t mem_idx = cpu_mmu_index(env, false);

#endif

    uintptr_t ra = GETPC();

    uint32_t fc = extract32(env->regs[0], 0, 8);

    uint32_t sc = extract32(env->regs[0], 8, 8);

    uint64_t pl = get_address(env, 1) & -16;

    uint64_t svh, svl;

    uint32_t cc;



    /* Sanity check the function code and storage characteristic.  */

    if (fc > 1 || sc > 3) {

        if (!s390_has_feat(S390_FEAT_COMPARE_AND_SWAP_AND_STORE_2)) {

            goto spec_exception;

        }

        if (fc > 2 || sc > 4 || (fc == 2 && (r3 & 1))) {

            goto spec_exception;

        }

    }



    /* Sanity check the alignments.  */

    if (extract32(a1, 0, 4 << fc) || extract32(a2, 0, 1 << sc)) {

        goto spec_exception;

    }



    /* Sanity check writability of the store address.  */

#ifndef CONFIG_USER_ONLY

    probe_write(env, a2, mem_idx, ra);

#endif



    /* Note that the compare-and-swap is atomic, and the store is atomic, but

       the complete operation is not.  Therefore we do not need to assert serial

       context in order to implement this.  That said, restart early if we can't

       support either operation that is supposed to be atomic.  */

    if (parallel_cpus) {

        int mask = 0;

#if !defined(CONFIG_ATOMIC64)

        mask = -8;

#elif !defined(CONFIG_ATOMIC128)

        mask = -16;

#endif

        if (((4 << fc) | (1 << sc)) & mask) {

            cpu_loop_exit_atomic(ENV_GET_CPU(env), ra);

        }

    }



    /* All loads happen before all stores.  For simplicity, load the entire

       store value area from the parameter list.  */

    svh = cpu_ldq_data_ra(env, pl + 16, ra);

    svl = cpu_ldq_data_ra(env, pl + 24, ra);



    switch (fc) {

    case 0:

        {

            uint32_t nv = cpu_ldl_data_ra(env, pl, ra);

            uint32_t cv = env->regs[r3];

            uint32_t ov;



            if (parallel_cpus) {

#ifdef CONFIG_USER_ONLY

                uint32_t *haddr = g2h(a1);

                ov = atomic_cmpxchg__nocheck(haddr, cv, nv);

#else

                TCGMemOpIdx oi = make_memop_idx(MO_TEUL | MO_ALIGN, mem_idx);

                ov = helper_atomic_cmpxchgl_be_mmu(env, a1, cv, nv, oi, ra);

#endif

            } else {

                ov = cpu_ldl_data_ra(env, a1, ra);

                cpu_stl_data_ra(env, a1, (ov == cv ? nv : ov), ra);

            }

            cc = (ov != cv);

            env->regs[r3] = deposit64(env->regs[r3], 32, 32, ov);

        }

        break;



    case 1:

        {

            uint64_t nv = cpu_ldq_data_ra(env, pl, ra);

            uint64_t cv = env->regs[r3];

            uint64_t ov;



            if (parallel_cpus) {

#ifdef CONFIG_ATOMIC64

# ifdef CONFIG_USER_ONLY

                uint64_t *haddr = g2h(a1);

                ov = atomic_cmpxchg__nocheck(haddr, cv, nv);

# else

                TCGMemOpIdx oi = make_memop_idx(MO_TEQ | MO_ALIGN, mem_idx);

                ov = helper_atomic_cmpxchgq_be_mmu(env, a1, cv, nv, oi, ra);

# endif

#else

                /* Note that we asserted !parallel_cpus above.  */

                g_assert_not_reached();

#endif

            } else {

                ov = cpu_ldq_data_ra(env, a1, ra);

                cpu_stq_data_ra(env, a1, (ov == cv ? nv : ov), ra);

            }

            cc = (ov != cv);

            env->regs[r3] = ov;

        }

        break;



    case 2:

        {

            uint64_t nvh = cpu_ldq_data_ra(env, pl, ra);

            uint64_t nvl = cpu_ldq_data_ra(env, pl + 8, ra);

            Int128 nv = int128_make128(nvl, nvh);

            Int128 cv = int128_make128(env->regs[r3 + 1], env->regs[r3]);

            Int128 ov;



            if (parallel_cpus) {

#ifdef CONFIG_ATOMIC128

                TCGMemOpIdx oi = make_memop_idx(MO_TEQ | MO_ALIGN_16, mem_idx);

                ov = helper_atomic_cmpxchgo_be_mmu(env, a1, cv, nv, oi, ra);

                cc = !int128_eq(ov, cv);

#else

                /* Note that we asserted !parallel_cpus above.  */

                g_assert_not_reached();

#endif

            } else {

                uint64_t oh = cpu_ldq_data_ra(env, a1 + 0, ra);

                uint64_t ol = cpu_ldq_data_ra(env, a1 + 8, ra);



                ov = int128_make128(ol, oh);

                cc = !int128_eq(ov, cv);

                if (cc) {

                    nv = ov;

                }



                cpu_stq_data_ra(env, a1 + 0, int128_gethi(nv), ra);

                cpu_stq_data_ra(env, a1 + 8, int128_getlo(nv), ra);

            }



            env->regs[r3 + 0] = int128_gethi(ov);

            env->regs[r3 + 1] = int128_getlo(ov);

        }

        break;



    default:

        g_assert_not_reached();

    }



    /* Store only if the comparison succeeded.  Note that above we use a pair

       of 64-bit big-endian loads, so for sc < 3 we must extract the value

       from the most-significant bits of svh.  */

    if (cc == 0) {

        switch (sc) {

        case 0:

            cpu_stb_data_ra(env, a2, svh >> 56, ra);

            break;

        case 1:

            cpu_stw_data_ra(env, a2, svh >> 48, ra);

            break;

        case 2:

            cpu_stl_data_ra(env, a2, svh >> 32, ra);

            break;

        case 3:

            cpu_stq_data_ra(env, a2, svh, ra);

            break;

        case 4:

            if (parallel_cpus) {

#ifdef CONFIG_ATOMIC128

                TCGMemOpIdx oi = make_memop_idx(MO_TEQ | MO_ALIGN_16, mem_idx);

                Int128 sv = int128_make128(svl, svh);

                helper_atomic_sto_be_mmu(env, a2, sv, oi, ra);

#else

                /* Note that we asserted !parallel_cpus above.  */

                g_assert_not_reached();

#endif

            } else {

                cpu_stq_data_ra(env, a2 + 0, svh, ra);

                cpu_stq_data_ra(env, a2 + 8, svl, ra);

            }

            break;

        default:

            g_assert_not_reached();

        }

    }



    return cc;



 spec_exception:

    cpu_restore_state(ENV_GET_CPU(env), ra);

    program_interrupt(env, PGM_SPECIFICATION, 6);

    g_assert_not_reached();

}
