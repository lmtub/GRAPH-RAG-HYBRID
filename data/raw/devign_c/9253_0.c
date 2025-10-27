void HELPER(cas2w)(CPUM68KState *env, uint32_t regs, uint32_t a1, uint32_t a2)

{

    uint32_t Dc1 = extract32(regs, 9, 3);

    uint32_t Dc2 = extract32(regs, 6, 3);

    uint32_t Du1 = extract32(regs, 3, 3);

    uint32_t Du2 = extract32(regs, 0, 3);

    int16_t c1 = env->dregs[Dc1];

    int16_t c2 = env->dregs[Dc2];

    int16_t u1 = env->dregs[Du1];

    int16_t u2 = env->dregs[Du2];

    int16_t l1, l2;

    uintptr_t ra = GETPC();



    if (parallel_cpus) {

        /* Tell the main loop we need to serialize this insn.  */

        cpu_loop_exit_atomic(ENV_GET_CPU(env), ra);

    } else {

        /* We're executing in a serial context -- no need to be atomic.  */

        l1 = cpu_lduw_data_ra(env, a1, ra);

        l2 = cpu_lduw_data_ra(env, a2, ra);

        if (l1 == c1 && l2 == c2) {

            cpu_stw_data_ra(env, a1, u1, ra);

            cpu_stw_data_ra(env, a2, u2, ra);

        }

    }



    if (c1 != l1) {

        env->cc_n = l1;

        env->cc_v = c1;

    } else {

        env->cc_n = l2;

        env->cc_v = c2;

    }

    env->cc_op = CC_OP_CMPW;

    env->dregs[Dc1] = deposit32(env->dregs[Dc1], 0, 16, l1);

    env->dregs[Dc2] = deposit32(env->dregs[Dc2], 0, 16, l2);

}
