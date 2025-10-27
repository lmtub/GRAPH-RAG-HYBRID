static uint32_t mvc_asc(CPUS390XState *env, int64_t l, uint64_t a1,

                        uint64_t mode1, uint64_t a2, uint64_t mode2)

{

    CPUState *cs = CPU(s390_env_get_cpu(env));

    target_ulong src, dest;

    int flags, cc = 0, i;



    if (!l) {

        return 0;

    } else if (l > 256) {

        /* max 256 */

        l = 256;

        cc = 3;

    }



    if (mmu_translate(env, a1, 1, mode1, &dest, &flags, true)) {

        cpu_loop_exit(CPU(s390_env_get_cpu(env)));

    }

    dest |= a1 & ~TARGET_PAGE_MASK;



    if (mmu_translate(env, a2, 0, mode2, &src, &flags, true)) {

        cpu_loop_exit(CPU(s390_env_get_cpu(env)));

    }

    src |= a2 & ~TARGET_PAGE_MASK;



    /* XXX replace w/ memcpy */

    for (i = 0; i < l; i++) {

        /* XXX be more clever */

        if ((((dest + i) & TARGET_PAGE_MASK) != (dest & TARGET_PAGE_MASK)) ||

            (((src + i) & TARGET_PAGE_MASK) != (src & TARGET_PAGE_MASK))) {

            mvc_asc(env, l - i, a1 + i, mode1, a2 + i, mode2);

            break;

        }

        stb_phys(cs->as, dest + i, ldub_phys(cs->as, src + i));

    }



    return cc;

}
