void HELPER(exception_return)(CPUARMState *env)

{

    int cur_el = arm_current_el(env);

    unsigned int spsr_idx = aarch64_banked_spsr_index(cur_el);

    uint32_t spsr = env->banked_spsr[spsr_idx];

    int new_el;



    aarch64_save_sp(env, cur_el);



    env->exclusive_addr = -1;



    /* We must squash the PSTATE.SS bit to zero unless both of the

     * following hold:

     *  1. debug exceptions are currently disabled

     *  2. singlestep will be active in the EL we return to

     * We check 1 here and 2 after we've done the pstate/cpsr write() to

     * transition to the EL we're going to.

     */

    if (arm_generate_debug_exceptions(env)) {

        spsr &= ~PSTATE_SS;

    }



    if (spsr & PSTATE_nRW) {

        /* TODO: We currently assume EL1/2/3 are running in AArch64.  */

        env->aarch64 = 0;

        new_el = 0;

        env->uncached_cpsr = 0x10;

        cpsr_write(env, spsr, ~0);

        if (!arm_singlestep_active(env)) {

            env->uncached_cpsr &= ~PSTATE_SS;

        }

        aarch64_sync_64_to_32(env);



        env->regs[15] = env->elr_el[1] & ~0x1;

    } else {

        new_el = extract32(spsr, 2, 2);

        if (new_el > cur_el

            || (new_el == 2 && !arm_feature(env, ARM_FEATURE_EL2))) {

            /* Disallow return to an EL which is unimplemented or higher

             * than the current one.

             */

            goto illegal_return;

        }

        if (extract32(spsr, 1, 1)) {

            /* Return with reserved M[1] bit set */

            goto illegal_return;

        }

        if (new_el == 0 && (spsr & PSTATE_SP)) {

            /* Return to EL0 with M[0] bit set */

            goto illegal_return;

        }

        env->aarch64 = 1;

        pstate_write(env, spsr);

        if (!arm_singlestep_active(env)) {

            env->pstate &= ~PSTATE_SS;

        }

        aarch64_restore_sp(env, new_el);

        env->pc = env->elr_el[cur_el];

    }



    return;



illegal_return:

    /* Illegal return events of various kinds have architecturally

     * mandated behaviour:

     * restore NZCV and DAIF from SPSR_ELx

     * set PSTATE.IL

     * restore PC from ELR_ELx

     * no change to exception level, execution state or stack pointer

     */

    env->pstate |= PSTATE_IL;

    env->pc = env->elr_el[cur_el];

    spsr &= PSTATE_NZCV | PSTATE_DAIF;

    spsr |= pstate_read(env) & ~(PSTATE_NZCV | PSTATE_DAIF);

    pstate_write(env, spsr);

    if (!arm_singlestep_active(env)) {

        env->pstate &= ~PSTATE_SS;

    }

}
