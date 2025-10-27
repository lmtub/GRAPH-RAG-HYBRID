uint32_t HELPER(v7m_mrs)(CPUARMState *env, uint32_t reg)

{

    uint32_t mask;

    unsigned el = arm_current_el(env);



    /* First handle registers which unprivileged can read */



    switch (reg) {

    case 0 ... 7: /* xPSR sub-fields */

        mask = 0;

        if ((reg & 1) && el) {

            mask |= XPSR_EXCP; /* IPSR (unpriv. reads as zero) */

        }

        if (!(reg & 4)) {

            mask |= XPSR_NZCV | XPSR_Q; /* APSR */

        }

        /* EPSR reads as zero */

        return xpsr_read(env) & mask;

        break;

    case 20: /* CONTROL */

        return env->v7m.control;

    }



    if (el == 0) {

        return 0; /* unprivileged reads others as zero */

    }



    switch (reg) {

    case 8: /* MSP */

        return (env->v7m.control & R_V7M_CONTROL_SPSEL_MASK) ?

            env->v7m.other_sp : env->regs[13];

    case 9: /* PSP */

        return (env->v7m.control & R_V7M_CONTROL_SPSEL_MASK) ?

            env->regs[13] : env->v7m.other_sp;

    case 16: /* PRIMASK */

        return env->v7m.primask[env->v7m.secure];

    case 17: /* BASEPRI */

    case 18: /* BASEPRI_MAX */

        return env->v7m.basepri[env->v7m.secure];

    case 19: /* FAULTMASK */

        return env->v7m.faultmask[env->v7m.secure];

    default:

        qemu_log_mask(LOG_GUEST_ERROR, "Attempt to read unknown special"

                                       " register %d\n", reg);

        return 0;

    }

}
