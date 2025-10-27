uint32_t HELPER(mvcl)(CPUS390XState *env, uint32_t r1, uint32_t r2)

{

    uintptr_t ra = GETPC();

    uint64_t destlen = env->regs[r1 + 1] & 0xffffff;

    uint64_t dest = get_address(env, r1);

    uint64_t srclen = env->regs[r2 + 1] & 0xffffff;

    uint64_t src = get_address(env, r2);

    uint8_t pad = env->regs[r2 + 1] >> 24;

    uint8_t v;

    uint32_t cc;



    if (destlen == srclen) {

        cc = 0;

    } else if (destlen < srclen) {

        cc = 1;

    } else {

        cc = 2;

    }



    if (srclen > destlen) {

        srclen = destlen;

    }



    for (; destlen && srclen; src++, dest++, destlen--, srclen--) {

        v = cpu_ldub_data_ra(env, src, ra);

        cpu_stb_data_ra(env, dest, v, ra);

    }



    for (; destlen; dest++, destlen--) {

        cpu_stb_data_ra(env, dest, pad, ra);

    }



    env->regs[r1 + 1] = destlen;

    /* can't use srclen here, we trunc'ed it */

    env->regs[r2 + 1] -= src - env->regs[r2];

    set_address(env, r1, dest);

    set_address(env, r2, src);



    return cc;

}
