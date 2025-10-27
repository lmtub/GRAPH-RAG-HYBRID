void HELPER(set_cp15)(CPUARMState *env, uint32_t insn, uint32_t val)

{

    int op1;

    int op2;

    int crm;



    op1 = (insn >> 21) & 7;

    op2 = (insn >> 5) & 7;

    crm = insn & 0xf;

    switch ((insn >> 16) & 0xf) {

    case 0:

        /* ID codes.  */

        if (arm_feature(env, ARM_FEATURE_XSCALE))

            break;

        if (arm_feature(env, ARM_FEATURE_OMAPCP))

            break;

        if (arm_feature(env, ARM_FEATURE_V7)

                && op1 == 2 && crm == 0 && op2 == 0) {

            env->cp15.c0_cssel = val & 0xf;

            break;

        }

        goto bad_reg;

    case 1: /* System configuration.  */

        if (arm_feature(env, ARM_FEATURE_V7)

                && op1 == 0 && crm == 1 && op2 == 0) {

            env->cp15.c1_scr = val;

            break;

        }

        if (arm_feature(env, ARM_FEATURE_OMAPCP))

            op2 = 0;

        switch (op2) {

        case 0:

            if (!arm_feature(env, ARM_FEATURE_XSCALE) || crm == 0)

                env->cp15.c1_sys = val;

            /* ??? Lots of these bits are not implemented.  */

            /* This may enable/disable the MMU, so do a TLB flush.  */

            tlb_flush(env, 1);

            break;

        case 1: /* Auxiliary control register.  */

            if (arm_feature(env, ARM_FEATURE_XSCALE)) {

                env->cp15.c1_xscaleauxcr = val;

                break;

            }

            /* Not implemented.  */

            break;

        case 2:

            if (arm_feature(env, ARM_FEATURE_XSCALE))

                goto bad_reg;

            if (env->cp15.c1_coproc != val) {

                env->cp15.c1_coproc = val;

                /* ??? Is this safe when called from within a TB?  */

                tb_flush(env);

            }

            break;

        default:

            goto bad_reg;

        }

        break;

    case 4: /* Reserved.  */

        goto bad_reg;

    case 6: /* MMU Fault address / MPU base/size.  */

        if (arm_feature(env, ARM_FEATURE_MPU)) {

            if (crm >= 8)

                goto bad_reg;

            env->cp15.c6_region[crm] = val;

        } else {

            if (arm_feature(env, ARM_FEATURE_OMAPCP))

                op2 = 0;

            switch (op2) {

            case 0:

                env->cp15.c6_data = val;

                break;

            case 1: /* ??? This is WFAR on armv6 */

            case 2:

                env->cp15.c6_insn = val;

                break;

            default:

                goto bad_reg;

            }

        }

        break;

    case 7: /* Cache control.  */

        env->cp15.c15_i_max = 0x000;

        env->cp15.c15_i_min = 0xff0;

        if (op1 != 0) {

            goto bad_reg;

        }

        break;

    case 9:

        if (arm_feature(env, ARM_FEATURE_OMAPCP))

            break;

        if (arm_feature(env, ARM_FEATURE_STRONGARM))

            break; /* Ignore ReadBuffer access */

        switch (crm) {

        case 0: /* Cache lockdown.  */

	    switch (op1) {

	    case 0: /* L1 cache.  */

		switch (op2) {

		case 0:

		    env->cp15.c9_data = val;

		    break;

		case 1:

		    env->cp15.c9_insn = val;

		    break;

		default:

		    goto bad_reg;

		}

		break;

	    case 1: /* L2 cache.  */

		/* Ignore writes to L2 lockdown/auxiliary registers.  */

		break;

	    default:

		goto bad_reg;

	    }

	    break;

        case 1: /* TCM memory region registers.  */

            /* Not implemented.  */

            goto bad_reg;

        default:

            goto bad_reg;

        }

        break;

    case 12: /* Reserved.  */

        goto bad_reg;

    }

    return;

bad_reg:

    /* ??? For debugging only.  Should raise illegal instruction exception.  */

    cpu_abort(env, "Unimplemented cp15 register write (c%d, c%d, {%d, %d})\n",

              (insn >> 16) & 0xf, crm, op1, op2);

}
