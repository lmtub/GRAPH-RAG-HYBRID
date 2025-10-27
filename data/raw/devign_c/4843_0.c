uint32_t HELPER(xc)(CPUS390XState *env, uint32_t l, uint64_t dest,

                    uint64_t src)

{

    int i;

    unsigned char x;

    uint32_t cc = 0;



    HELPER_LOG("%s l %d dest %" PRIx64 " src %" PRIx64 "\n",

               __func__, l, dest, src);



#ifndef CONFIG_USER_ONLY

    /* xor with itself is the same as memset(0) */

    if ((l > 32) && (src == dest) &&

        (src & TARGET_PAGE_MASK) == ((src + l) & TARGET_PAGE_MASK)) {

        mvc_fast_memset(env, l + 1, dest, 0);

        return 0;

    }

#else

    if (src == dest) {

        memset(g2h(dest), 0, l + 1);

        return 0;

    }

#endif



    for (i = 0; i <= l; i++) {

        x = cpu_ldub_data(env, dest + i) ^ cpu_ldub_data(env, src + i);

        if (x) {

            cc = 1;

        }

        cpu_stb_data(env, dest + i, x);

    }

    return cc;

}
