void HELPER(set_cp15)(CPUState *env, uint32_t insn, uint32_t val)

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

        case 1: /* Auxiliary cotrol register.  */

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

    case 2: /* MMU Page table control / MPU cache control.  */

        if (arm_feature(env, ARM_FEATURE_MPU)) {

            switch (op2) {

            case 0:

                env->cp15.c2_data = val;

                break;

            case 1:

                env->cp15.c2_insn = val;

                break;

            default:

                goto bad_reg;

            }

        } else {

	    switch (op2) {

	    case 0:

		env->cp15.c2_base0 = val;

		break;

	    case 1:

		env->cp15.c2_base1 = val;

		break;

	    case 2:

                val &= 7;

                env->cp15.c2_control = val;

		env->cp15.c2_mask = ~(((uint32_t)0xffffffffu) >> val);

                env->cp15.c2_base_mask = ~((uint32_t)0x3fffu >> val);

		break;

	    default:

		goto bad_reg;

	    }

        }

        break;

    case 3: /* MMU Domain access control / MPU write buffer control.  */

        env->cp15.c3 = val;

        tlb_flush(env, 1); /* Flush TLB as domain not tracked in TLB */

        break;

    case 4: /* Reserved.  */

        goto bad_reg;

    case 5: /* MMU Fault status / MPU access permission.  */

        if (arm_feature(env, ARM_FEATURE_OMAPCP))

            op2 = 0;

        switch (op2) {

        case 0:

            if (arm_feature(env, ARM_FEATURE_MPU))

                val = extended_mpu_ap_bits(val);

            env->cp15.c5_data = val;

            break;

        case 1:

            if (arm_feature(env, ARM_FEATURE_MPU))

                val = extended_mpu_ap_bits(val);

            env->cp15.c5_insn = val;

            break;

        case 2:

            if (!arm_feature(env, ARM_FEATURE_MPU))

                goto bad_reg;

            env->cp15.c5_data = val;

            break;

        case 3:

            if (!arm_feature(env, ARM_FEATURE_MPU))

                goto bad_reg;

            env->cp15.c5_insn = val;

            break;

        default:

            goto bad_reg;

        }

        break;

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

        /* No cache, so nothing to do.  */

        /* ??? MPCore has VA to PA translation functions.  */

        break;

    case 8: /* MMU TLB control.  */

        switch (op2) {

        case 0: /* Invalidate all.  */

            tlb_flush(env, 0);

            break;

        case 1: /* Invalidate single TLB entry.  */

#if 0

            /* ??? This is wrong for large pages and sections.  */

            /* As an ugly hack to make linux work we always flush a 4K

               pages.  */

            val &= 0xfffff000;

            tlb_flush_page(env, val);

            tlb_flush_page(env, val + 0x400);

            tlb_flush_page(env, val + 0x800);

            tlb_flush_page(env, val + 0xc00);

#else

            tlb_flush(env, 1);

#endif

            break;

        case 2: /* Invalidate on ASID.  */

            tlb_flush(env, val == 0);

            break;

        case 3: /* Invalidate single entry on MVA.  */

            /* ??? This is like case 1, but ignores ASID.  */

            tlb_flush(env, 1);

            break;

        default:

            goto bad_reg;

        }

        break;

    case 9:

        if (arm_feature(env, ARM_FEATURE_OMAPCP))

            break;

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

    case 10: /* MMU TLB lockdown.  */

        /* ??? TLB lockdown not implemented.  */

        break;

    case 12: /* Reserved.  */

        goto bad_reg;

    case 13: /* Process ID.  */

        switch (op2) {

        case 0:

            /* Unlike real hardware the qemu TLB uses virtual addresses,

               not modified virtual addresses, so this causes a TLB flush.

             */

            if (env->cp15.c13_fcse != val)

              tlb_flush(env, 1);

            env->cp15.c13_fcse = val;

            break;

        case 1:

            /* This changes the ASID, so do a TLB flush.  */

            if (env->cp15.c13_context != val

                && !arm_feature(env, ARM_FEATURE_MPU))

              tlb_flush(env, 0);

            env->cp15.c13_context = val;

            break;

        case 2:

            env->cp15.c13_tls1 = val;

            break;

        case 3:

            env->cp15.c13_tls2 = val;

            break;

        case 4:

            env->cp15.c13_tls3 = val;

            break;

        default:

            goto bad_reg;

        }

        break;

    case 14: /* Reserved.  */

        goto bad_reg;

    case 15: /* Implementation specific.  */

        if (arm_feature(env, ARM_FEATURE_XSCALE)) {

            if (op2 == 0 && crm == 1) {

                if (env->cp15.c15_cpar != (val & 0x3fff)) {

                    /* Changes cp0 to cp13 behavior, so needs a TB flush.  */

                    tb_flush(env);

                    env->cp15.c15_cpar = val & 0x3fff;

                }

                break;

            }

            goto bad_reg;

        }

        if (arm_feature(env, ARM_FEATURE_OMAPCP)) {

            switch (crm) {

            case 0:

                break;

            case 1: /* Set TI925T configuration.  */

                env->cp15.c15_ticonfig = val & 0xe7;

                env->cp15.c0_cpuid = (val & (1 << 5)) ? /* OS_TYPE bit */

                        ARM_CPUID_TI915T : ARM_CPUID_TI925T;

                break;

            case 2: /* Set I_max.  */

                env->cp15.c15_i_max = val;

                break;

            case 3: /* Set I_min.  */

                env->cp15.c15_i_min = val;

                break;

            case 4: /* Set thread-ID.  */

                env->cp15.c15_threadid = val & 0xffff;

                break;

            case 8: /* Wait-for-interrupt (deprecated).  */

                cpu_interrupt(env, CPU_INTERRUPT_HALT);

                break;

            default:

                goto bad_reg;

            }

        }

        break;

    }

    return;

bad_reg:

    /* ??? For debugging only.  Should raise illegal instruction exception.  */

    cpu_abort(env, "Unimplemented cp15 register write (c%d, c%d, {%d, %d})\n",

              (insn >> 16) & 0xf, crm, op1, op2);

}
