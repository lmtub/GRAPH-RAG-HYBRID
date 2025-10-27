void HELPER(v7m_msr)(CPUARMState *env, uint32_t maskreg, uint32_t val)

{

    /* We're passed bits [11..0] of the instruction; extract

     * SYSm and the mask bits.

     * Invalid combinations of SYSm and mask are UNPREDICTABLE;

     * we choose to treat them as if the mask bits were valid.

     * NB that the pseudocode 'mask' variable is bits [11..10],

     * whereas ours is [11..8].

     */

    uint32_t mask = extract32(maskreg, 8, 4);

    uint32_t reg = extract32(maskreg, 0, 8);



    if (arm_current_el(env) == 0 && reg > 7) {

        /* only xPSR sub-fields may be written by unprivileged */

        return;

    }



    switch (reg) {

    case 0 ... 7: /* xPSR sub-fields */

        /* only APSR is actually writable */

        if (!(reg & 4)) {

            uint32_t apsrmask = 0;



            if (mask & 8) {

                apsrmask |= 0xf8000000; /* APSR NZCVQ */

            }

            if ((mask & 4) && arm_feature(env, ARM_FEATURE_THUMB_DSP)) {

                apsrmask |= 0x000f0000; /* APSR GE[3:0] */

            }

            xpsr_write(env, val, apsrmask);

        }

        break;

    case 8: /* MSP */

        if (env->v7m.control & R_V7M_CONTROL_SPSEL_MASK) {

            env->v7m.other_sp = val;

        } else {

            env->regs[13] = val;

        }

        break;

    case 9: /* PSP */

        if (env->v7m.control & R_V7M_CONTROL_SPSEL_MASK) {

            env->regs[13] = val;

        } else {

            env->v7m.other_sp = val;

        }

        break;

    case 16: /* PRIMASK */

        if (val & 1) {

            env->daif |= PSTATE_I;

        } else {

            env->daif &= ~PSTATE_I;

        }

        break;

    case 17: /* BASEPRI */

        env->v7m.basepri = val & 0xff;

        break;

    case 18: /* BASEPRI_MAX */

        val &= 0xff;

        if (val != 0 && (val < env->v7m.basepri || env->v7m.basepri == 0))

            env->v7m.basepri = val;

        break;

    case 19: /* FAULTMASK */

        if (val & 1) {

            env->daif |= PSTATE_F;

        } else {

            env->daif &= ~PSTATE_F;

        }

        break;

    case 20: /* CONTROL */

        switch_v7m_sp(env, (val & R_V7M_CONTROL_SPSEL_MASK) != 0);

        env->v7m.control = val & (R_V7M_CONTROL_SPSEL_MASK |

                                  R_V7M_CONTROL_NPRIV_MASK);

        break;

    default:

        qemu_log_mask(LOG_GUEST_ERROR, "Attempt to write unknown special"

                                       " register %d\n", reg);

        return;

    }

}
