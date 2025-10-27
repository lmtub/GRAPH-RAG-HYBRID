uint32_t HELPER(get_cp15)(CPUARMState *env, uint32_t insn)

{

    int op1;

    int op2;

    int crm;



    op1 = (insn >> 21) & 7;

    op2 = (insn >> 5) & 7;

    crm = insn & 0xf;

    switch ((insn >> 16) & 0xf) {

    case 0: /* ID codes.  */

        switch (op1) {

        case 0:

            switch (crm) {

            case 0:

                switch (op2) {

                case 0: /* Device ID.  */

                    return env->cp15.c0_cpuid;

                case 1: /* Cache Type.  */

		    return env->cp15.c0_cachetype;

                case 2: /* TCM status.  */

                    return 0;

                case 3: /* TLB type register.  */

                    return 0; /* No lockable TLB entries.  */

                case 5: /* MPIDR */

                    /* The MPIDR was standardised in v7; prior to

                     * this it was implemented only in the 11MPCore.

                     * For all other pre-v7 cores it does not exist.

                     */

                    if (arm_feature(env, ARM_FEATURE_V7) ||

                        ARM_CPUID(env) == ARM_CPUID_ARM11MPCORE) {

                        int mpidr = env->cpu_index;

                        /* We don't support setting cluster ID ([8..11])

                         * so these bits always RAZ.

                         */

                        if (arm_feature(env, ARM_FEATURE_V7MP)) {

                            mpidr |= (1 << 31);

                            /* Cores which are uniprocessor (non-coherent)

                             * but still implement the MP extensions set

                             * bit 30. (For instance, A9UP.) However we do

                             * not currently model any of those cores.

                             */

                        }

                        return mpidr;

                    }

                    /* otherwise fall through to the unimplemented-reg case */

                default:

                    goto bad_reg;

                }

            case 3: case 4: case 5: case 6: case 7:

                return 0;

            default:

                goto bad_reg;

            }

        default:

            goto bad_reg;

        }

    case 4: /* Reserved.  */

        goto bad_reg;

    case 11: /* TCM DMA control.  */

    case 12: /* Reserved.  */

        goto bad_reg;

    }

bad_reg:

    /* ??? For debugging only.  Should raise illegal instruction exception.  */

    cpu_abort(env, "Unimplemented cp15 register read (c%d, c%d, {%d, %d})\n",

              (insn >> 16) & 0xf, crm, op1, op2);

    return 0;

}
