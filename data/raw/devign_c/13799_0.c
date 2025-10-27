void HELPER(srst)(CPUS390XState *env, uint32_t r1, uint32_t r2)

{

    uintptr_t ra = GETPC();

    uint64_t end, str;

    uint32_t len;

    uint8_t v, c = env->regs[0];



    /* Bits 32-55 must contain all 0.  */

    if (env->regs[0] & 0xffffff00u) {

        cpu_restore_state(ENV_GET_CPU(env), ra);

        program_interrupt(env, PGM_SPECIFICATION, 6);

    }



    str = get_address(env, r2);

    end = get_address(env, r1);



    /* Lest we fail to service interrupts in a timely manner, limit the

       amount of work we're willing to do.  For now, let's cap at 8k.  */

    for (len = 0; len < 0x2000; ++len) {

        if (str + len == end) {

            /* Character not found.  R1 & R2 are unmodified.  */

            env->cc_op = 2;

            return;

        }

        v = cpu_ldub_data_ra(env, str + len, ra);

        if (v == c) {

            /* Character found.  Set R1 to the location; R2 is unmodified.  */

            env->cc_op = 1;

            set_address(env, r1, str + len);

            return;

        }

    }



    /* CPU-determined bytes processed.  Advance R2 to next byte to process.  */

    env->cc_op = 3;

    set_address(env, r2, str + len);

}
