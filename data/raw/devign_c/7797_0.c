static int get_cpsr(QEMUFile *f, void *opaque, size_t size,

                    VMStateField *field)

{

    ARMCPU *cpu = opaque;

    CPUARMState *env = &cpu->env;

    uint32_t val = qemu_get_be32(f);



    if (arm_feature(env, ARM_FEATURE_M)) {

        if (val & XPSR_EXCP) {

            /* This is a CPSR format value from an older QEMU. (We can tell

             * because values transferred in XPSR format always have zero

             * for the EXCP field, and CPSR format will always have bit 4

             * set in CPSR_M.) Rearrange it into XPSR format. The significant

             * differences are that the T bit is not in the same place, the

             * primask/faultmask info may be in the CPSR I and F bits, and

             * we do not want the mode bits.

             */

            uint32_t newval = val;



            newval &= (CPSR_NZCV | CPSR_Q | CPSR_IT | CPSR_GE);

            if (val & CPSR_T) {

                newval |= XPSR_T;

            }

            /* If the I or F bits are set then this is a migration from

             * an old QEMU which still stored the M profile FAULTMASK

             * and PRIMASK in env->daif. For a new QEMU, the data is

             * transferred using the vmstate_m_faultmask_primask subsection.

             */

            if (val & CPSR_F) {

                env->v7m.faultmask = 1;

            }

            if (val & CPSR_I) {

                env->v7m.primask = 1;

            }

            val = newval;

        }

        /* Ignore the low bits, they are handled by vmstate_m. */

        xpsr_write(env, val, ~XPSR_EXCP);

        return 0;

    }



    env->aarch64 = ((val & PSTATE_nRW) == 0);



    if (is_a64(env)) {

        pstate_write(env, val);

        return 0;

    }



    cpsr_write(env, val, 0xffffffff, CPSRWriteRaw);

    return 0;

}
