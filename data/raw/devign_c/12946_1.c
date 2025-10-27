void HELPER(mvc)(CPUS390XState *env, uint32_t l, uint64_t dest, uint64_t src)

{

    int i = 0;

    int x = 0;

    uint32_t l_64 = (l + 1) / 8;



    HELPER_LOG("%s l %d dest %" PRIx64 " src %" PRIx64 "\n",

               __func__, l, dest, src);



#ifndef CONFIG_USER_ONLY

    if ((l > 32) &&

        (src & TARGET_PAGE_MASK) == ((src + l) & TARGET_PAGE_MASK) &&

        (dest & TARGET_PAGE_MASK) == ((dest + l) & TARGET_PAGE_MASK)) {

        if (dest == (src + 1)) {

            mvc_fast_memset(env, l + 1, dest, cpu_ldub_data(env, src));

            return;

        } else if ((src & TARGET_PAGE_MASK) != (dest & TARGET_PAGE_MASK)) {

            mvc_fast_memmove(env, l + 1, dest, src);

            return;

        }

    }

#else

    if (dest == (src + 1)) {

        memset(g2h(dest), cpu_ldub_data(env, src), l + 1);

        return;

    } else {

        memmove(g2h(dest), g2h(src), l + 1);

        return;

    }

#endif



    /* handle the parts that fit into 8-byte loads/stores */

    if (dest != (src + 1)) {

        for (i = 0; i < l_64; i++) {

            cpu_stq_data(env, dest + x, cpu_ldq_data(env, src + x));

            x += 8;

        }

    }



    /* slow version crossing pages with byte accesses */

    for (i = x; i <= l; i++) {

        cpu_stb_data(env, dest + i, cpu_ldub_data(env, src + i));

    }

}
