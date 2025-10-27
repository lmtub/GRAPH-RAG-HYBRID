static uint32_t nvic_readl(NVICState *s, uint32_t offset, MemTxAttrs attrs)

{

    ARMCPU *cpu = s->cpu;

    uint32_t val;



    switch (offset) {

    case 4: /* Interrupt Control Type.  */

        return ((s->num_irq - NVIC_FIRST_IRQ) / 32) - 1;

    case 0xd00: /* CPUID Base.  */

        return cpu->midr;

    case 0xd04: /* Interrupt Control State.  */

        /* VECTACTIVE */

        val = cpu->env.v7m.exception;

        /* VECTPENDING */

        val |= (s->vectpending & 0xff) << 12;

        /* ISRPENDING - set if any external IRQ is pending */

        if (nvic_isrpending(s)) {

            val |= (1 << 22);

        }

        /* RETTOBASE - set if only one handler is active */

        if (nvic_rettobase(s)) {

            val |= (1 << 11);

        }

        /* PENDSTSET */

        if (s->vectors[ARMV7M_EXCP_SYSTICK].pending) {

            val |= (1 << 26);

        }

        /* PENDSVSET */

        if (s->vectors[ARMV7M_EXCP_PENDSV].pending) {

            val |= (1 << 28);

        }

        /* NMIPENDSET */

        if (s->vectors[ARMV7M_EXCP_NMI].pending) {

            val |= (1 << 31);

        }

        /* ISRPREEMPT not implemented */

        return val;

    case 0xd08: /* Vector Table Offset.  */

        return cpu->env.v7m.vecbase[attrs.secure];

    case 0xd0c: /* Application Interrupt/Reset Control.  */

        return 0xfa050000 | (s->prigroup << 8);

    case 0xd10: /* System Control.  */

        /* TODO: Implement SLEEPONEXIT.  */

        return 0;

    case 0xd14: /* Configuration Control.  */

        return cpu->env.v7m.ccr;

    case 0xd24: /* System Handler Status.  */

        val = 0;

        if (s->vectors[ARMV7M_EXCP_MEM].active) {

            val |= (1 << 0);

        }

        if (s->vectors[ARMV7M_EXCP_BUS].active) {

            val |= (1 << 1);

        }

        if (s->vectors[ARMV7M_EXCP_USAGE].active) {

            val |= (1 << 3);

        }

        if (s->vectors[ARMV7M_EXCP_SVC].active) {

            val |= (1 << 7);

        }

        if (s->vectors[ARMV7M_EXCP_DEBUG].active) {

            val |= (1 << 8);

        }

        if (s->vectors[ARMV7M_EXCP_PENDSV].active) {

            val |= (1 << 10);

        }

        if (s->vectors[ARMV7M_EXCP_SYSTICK].active) {

            val |= (1 << 11);

        }

        if (s->vectors[ARMV7M_EXCP_USAGE].pending) {

            val |= (1 << 12);

        }

        if (s->vectors[ARMV7M_EXCP_MEM].pending) {

            val |= (1 << 13);

        }

        if (s->vectors[ARMV7M_EXCP_BUS].pending) {

            val |= (1 << 14);

        }

        if (s->vectors[ARMV7M_EXCP_SVC].pending) {

            val |= (1 << 15);

        }

        if (s->vectors[ARMV7M_EXCP_MEM].enabled) {

            val |= (1 << 16);

        }

        if (s->vectors[ARMV7M_EXCP_BUS].enabled) {

            val |= (1 << 17);

        }

        if (s->vectors[ARMV7M_EXCP_USAGE].enabled) {

            val |= (1 << 18);

        }

        return val;

    case 0xd28: /* Configurable Fault Status.  */

        return cpu->env.v7m.cfsr;

    case 0xd2c: /* Hard Fault Status.  */

        return cpu->env.v7m.hfsr;

    case 0xd30: /* Debug Fault Status.  */

        return cpu->env.v7m.dfsr;

    case 0xd34: /* MMFAR MemManage Fault Address */

        return cpu->env.v7m.mmfar;

    case 0xd38: /* Bus Fault Address.  */

        return cpu->env.v7m.bfar;

    case 0xd3c: /* Aux Fault Status.  */

        /* TODO: Implement fault status registers.  */

        qemu_log_mask(LOG_UNIMP,

                      "Aux Fault status registers unimplemented\n");

        return 0;

    case 0xd40: /* PFR0.  */

        return 0x00000030;

    case 0xd44: /* PRF1.  */

        return 0x00000200;

    case 0xd48: /* DFR0.  */

        return 0x00100000;

    case 0xd4c: /* AFR0.  */

        return 0x00000000;

    case 0xd50: /* MMFR0.  */

        return 0x00000030;

    case 0xd54: /* MMFR1.  */

        return 0x00000000;

    case 0xd58: /* MMFR2.  */

        return 0x00000000;

    case 0xd5c: /* MMFR3.  */

        return 0x00000000;

    case 0xd60: /* ISAR0.  */

        return 0x01141110;

    case 0xd64: /* ISAR1.  */

        return 0x02111000;

    case 0xd68: /* ISAR2.  */

        return 0x21112231;

    case 0xd6c: /* ISAR3.  */

        return 0x01111110;

    case 0xd70: /* ISAR4.  */

        return 0x01310102;

    /* TODO: Implement debug registers.  */

    case 0xd90: /* MPU_TYPE */

        /* Unified MPU; if the MPU is not present this value is zero */

        return cpu->pmsav7_dregion << 8;

        break;

    case 0xd94: /* MPU_CTRL */

        return cpu->env.v7m.mpu_ctrl;

    case 0xd98: /* MPU_RNR */

        return cpu->env.pmsav7.rnr[attrs.secure];

    case 0xd9c: /* MPU_RBAR */

    case 0xda4: /* MPU_RBAR_A1 */

    case 0xdac: /* MPU_RBAR_A2 */

    case 0xdb4: /* MPU_RBAR_A3 */

    {

        int region = cpu->env.pmsav7.rnr[attrs.secure];



        if (arm_feature(&cpu->env, ARM_FEATURE_V8)) {

            /* PMSAv8M handling of the aliases is different from v7M:

             * aliases A1, A2, A3 override the low two bits of the region

             * number in MPU_RNR, and there is no 'region' field in the

             * RBAR register.

             */

            int aliasno = (offset - 0xd9c) / 8; /* 0..3 */

            if (aliasno) {

                region = deposit32(region, 0, 2, aliasno);

            }

            if (region >= cpu->pmsav7_dregion) {

                return 0;

            }

            return cpu->env.pmsav8.rbar[attrs.secure][region];

        }



        if (region >= cpu->pmsav7_dregion) {

            return 0;

        }

        return (cpu->env.pmsav7.drbar[region] & 0x1f) | (region & 0xf);

    }

    case 0xda0: /* MPU_RASR (v7M), MPU_RLAR (v8M) */

    case 0xda8: /* MPU_RASR_A1 (v7M), MPU_RLAR_A1 (v8M) */

    case 0xdb0: /* MPU_RASR_A2 (v7M), MPU_RLAR_A2 (v8M) */

    case 0xdb8: /* MPU_RASR_A3 (v7M), MPU_RLAR_A3 (v8M) */

    {

        int region = cpu->env.pmsav7.rnr[attrs.secure];



        if (arm_feature(&cpu->env, ARM_FEATURE_V8)) {

            /* PMSAv8M handling of the aliases is different from v7M:

             * aliases A1, A2, A3 override the low two bits of the region

             * number in MPU_RNR.

             */

            int aliasno = (offset - 0xda0) / 8; /* 0..3 */

            if (aliasno) {

                region = deposit32(region, 0, 2, aliasno);

            }

            if (region >= cpu->pmsav7_dregion) {

                return 0;

            }

            return cpu->env.pmsav8.rlar[attrs.secure][region];

        }



        if (region >= cpu->pmsav7_dregion) {

            return 0;

        }

        return ((cpu->env.pmsav7.dracr[region] & 0xffff) << 16) |

            (cpu->env.pmsav7.drsr[region] & 0xffff);

    }

    case 0xdc0: /* MPU_MAIR0 */

        if (!arm_feature(&cpu->env, ARM_FEATURE_V8)) {

            goto bad_offset;

        }

        return cpu->env.pmsav8.mair0[attrs.secure];

    case 0xdc4: /* MPU_MAIR1 */

        if (!arm_feature(&cpu->env, ARM_FEATURE_V8)) {

            goto bad_offset;

        }

        return cpu->env.pmsav8.mair1[attrs.secure];

    default:

    bad_offset:

        qemu_log_mask(LOG_GUEST_ERROR, "NVIC: Bad read offset 0x%x\n", offset);

        return 0;

    }

}
