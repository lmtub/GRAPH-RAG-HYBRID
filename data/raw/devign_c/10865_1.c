static uint32_t nvic_readl(NVICState *s, uint32_t offset, MemTxAttrs attrs)

{

    ARMCPU *cpu = s->cpu;

    uint32_t val;



    switch (offset) {

    case 4: /* Interrupt Control Type.  */

        return ((s->num_irq - NVIC_FIRST_IRQ) / 32) - 1;

    case 0x380 ... 0x3bf: /* NVIC_ITNS<n> */

    {

        int startvec = 32 * (offset - 0x380) + NVIC_FIRST_IRQ;

        int i;



        if (!arm_feature(&cpu->env, ARM_FEATURE_V8)) {

            goto bad_offset;

        }

        if (!attrs.secure) {

            return 0;

        }

        val = 0;

        for (i = 0; i < 32 && startvec + i < s->num_irq; i++) {

            if (s->itns[startvec + i]) {

                val |= (1 << i);

            }

        }

        return val;

    }

    case 0xd00: /* CPUID Base.  */

        return cpu->midr;

    case 0xd04: /* Interrupt Control State (ICSR) */

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

        if (attrs.secure) {

            /* PENDSTSET */

            if (s->sec_vectors[ARMV7M_EXCP_SYSTICK].pending) {

                val |= (1 << 26);

            }

            /* PENDSVSET */

            if (s->sec_vectors[ARMV7M_EXCP_PENDSV].pending) {

                val |= (1 << 28);

            }

        } else {

            /* PENDSTSET */

            if (s->vectors[ARMV7M_EXCP_SYSTICK].pending) {

                val |= (1 << 26);

            }

            /* PENDSVSET */

            if (s->vectors[ARMV7M_EXCP_PENDSV].pending) {

                val |= (1 << 28);

            }

        }

        /* NMIPENDSET */

        if ((cpu->env.v7m.aircr & R_V7M_AIRCR_BFHFNMINS_MASK) &&

            s->vectors[ARMV7M_EXCP_NMI].pending) {

            val |= (1 << 31);

        }

        /* ISRPREEMPT: RES0 when halting debug not implemented */

        /* STTNS: RES0 for the Main Extension */

        return val;

    case 0xd08: /* Vector Table Offset.  */

        return cpu->env.v7m.vecbase[attrs.secure];

    case 0xd0c: /* Application Interrupt/Reset Control (AIRCR) */

        val = 0xfa050000 | (s->prigroup[attrs.secure] << 8);

        if (attrs.secure) {

            /* s->aircr stores PRIS, BFHFNMINS, SYSRESETREQS */

            val |= cpu->env.v7m.aircr;

        } else {

            if (arm_feature(&cpu->env, ARM_FEATURE_V8)) {

                /* BFHFNMINS is R/O from NS; other bits are RAZ/WI. If

                 * security isn't supported then BFHFNMINS is RAO (and

                 * the bit in env.v7m.aircr is always set).

                 */

                val |= cpu->env.v7m.aircr & R_V7M_AIRCR_BFHFNMINS_MASK;

            }

        }

        return val;

    case 0xd10: /* System Control.  */

        /* TODO: Implement SLEEPONEXIT.  */

        return 0;

    case 0xd14: /* Configuration Control.  */

        /* The BFHFNMIGN bit is the only non-banked bit; we

         * keep it in the non-secure copy of the register.

         */

        val = cpu->env.v7m.ccr[attrs.secure];

        val |= cpu->env.v7m.ccr[M_REG_NS] & R_V7M_CCR_BFHFNMIGN_MASK;

        return val;

    case 0xd24: /* System Handler Control and State (SHCSR) */

        val = 0;

        if (attrs.secure) {

            if (s->sec_vectors[ARMV7M_EXCP_MEM].active) {

                val |= (1 << 0);

            }

            if (s->sec_vectors[ARMV7M_EXCP_HARD].active) {

                val |= (1 << 2);

            }

            if (s->sec_vectors[ARMV7M_EXCP_USAGE].active) {

                val |= (1 << 3);

            }

            if (s->sec_vectors[ARMV7M_EXCP_SVC].active) {

                val |= (1 << 7);

            }

            if (s->sec_vectors[ARMV7M_EXCP_PENDSV].active) {

                val |= (1 << 10);

            }

            if (s->sec_vectors[ARMV7M_EXCP_SYSTICK].active) {

                val |= (1 << 11);

            }

            if (s->sec_vectors[ARMV7M_EXCP_USAGE].pending) {

                val |= (1 << 12);

            }

            if (s->sec_vectors[ARMV7M_EXCP_MEM].pending) {

                val |= (1 << 13);

            }

            if (s->sec_vectors[ARMV7M_EXCP_SVC].pending) {

                val |= (1 << 15);

            }

            if (s->sec_vectors[ARMV7M_EXCP_MEM].enabled) {

                val |= (1 << 16);

            }

            if (s->sec_vectors[ARMV7M_EXCP_USAGE].enabled) {

                val |= (1 << 18);

            }

            if (s->sec_vectors[ARMV7M_EXCP_HARD].pending) {

                val |= (1 << 21);

            }

            /* SecureFault is not banked but is always RAZ/WI to NS */

            if (s->vectors[ARMV7M_EXCP_SECURE].active) {

                val |= (1 << 4);

            }

            if (s->vectors[ARMV7M_EXCP_SECURE].enabled) {

                val |= (1 << 19);

            }

            if (s->vectors[ARMV7M_EXCP_SECURE].pending) {

                val |= (1 << 20);

            }

        } else {

            if (s->vectors[ARMV7M_EXCP_MEM].active) {

                val |= (1 << 0);

            }

            if (arm_feature(&cpu->env, ARM_FEATURE_V8)) {

                /* HARDFAULTACT, HARDFAULTPENDED not present in v7M */

                if (s->vectors[ARMV7M_EXCP_HARD].active) {

                    val |= (1 << 2);

                }

                if (s->vectors[ARMV7M_EXCP_HARD].pending) {

                    val |= (1 << 21);

                }

            }

            if (s->vectors[ARMV7M_EXCP_USAGE].active) {

                val |= (1 << 3);

            }

            if (s->vectors[ARMV7M_EXCP_SVC].active) {

                val |= (1 << 7);

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

            if (s->vectors[ARMV7M_EXCP_SVC].pending) {

                val |= (1 << 15);

            }

            if (s->vectors[ARMV7M_EXCP_MEM].enabled) {

                val |= (1 << 16);

            }

            if (s->vectors[ARMV7M_EXCP_USAGE].enabled) {

                val |= (1 << 18);

            }

        }

        if (attrs.secure || (cpu->env.v7m.aircr & R_V7M_AIRCR_BFHFNMINS_MASK)) {

            if (s->vectors[ARMV7M_EXCP_BUS].active) {

                val |= (1 << 1);

            }

            if (s->vectors[ARMV7M_EXCP_BUS].pending) {

                val |= (1 << 14);

            }

            if (s->vectors[ARMV7M_EXCP_BUS].enabled) {

                val |= (1 << 17);

            }

            if (arm_feature(&cpu->env, ARM_FEATURE_V8) &&

                s->vectors[ARMV7M_EXCP_NMI].active) {

                /* NMIACT is not present in v7M */

                val |= (1 << 5);

            }

        }



        /* TODO: this is RAZ/WI from NS if DEMCR.SDME is set */

        if (s->vectors[ARMV7M_EXCP_DEBUG].active) {

            val |= (1 << 8);

        }

        return val;

    case 0xd28: /* Configurable Fault Status.  */

        /* The BFSR bits [15:8] are shared between security states

         * and we store them in the NS copy

         */

        val = cpu->env.v7m.cfsr[attrs.secure];

        val |= cpu->env.v7m.cfsr[M_REG_NS] & R_V7M_CFSR_BFSR_MASK;

        return val;

    case 0xd2c: /* Hard Fault Status.  */

        return cpu->env.v7m.hfsr;

    case 0xd30: /* Debug Fault Status.  */

        return cpu->env.v7m.dfsr;

    case 0xd34: /* MMFAR MemManage Fault Address */

        return cpu->env.v7m.mmfar[attrs.secure];

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

        return cpu->env.v7m.mpu_ctrl[attrs.secure];

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

    case 0xdd0: /* SAU_CTRL */

        if (!arm_feature(&cpu->env, ARM_FEATURE_V8)) {

            goto bad_offset;

        }

        if (!attrs.secure) {

            return 0;

        }

        return cpu->env.sau.ctrl;

    case 0xdd4: /* SAU_TYPE */

        if (!arm_feature(&cpu->env, ARM_FEATURE_V8)) {

            goto bad_offset;

        }

        if (!attrs.secure) {

            return 0;

        }

        return cpu->sau_sregion;

    case 0xdd8: /* SAU_RNR */

        if (!arm_feature(&cpu->env, ARM_FEATURE_V8)) {

            goto bad_offset;

        }

        if (!attrs.secure) {

            return 0;

        }

        return cpu->env.sau.rnr;

    case 0xddc: /* SAU_RBAR */

    {

        int region = cpu->env.sau.rnr;



        if (!arm_feature(&cpu->env, ARM_FEATURE_V8)) {

            goto bad_offset;

        }

        if (!attrs.secure) {

            return 0;

        }

        if (region >= cpu->sau_sregion) {

            return 0;

        }

        return cpu->env.sau.rbar[region];

    }

    case 0xde0: /* SAU_RLAR */

    {

        int region = cpu->env.sau.rnr;



        if (!arm_feature(&cpu->env, ARM_FEATURE_V8)) {

            goto bad_offset;

        }

        if (!attrs.secure) {

            return 0;

        }

        if (region >= cpu->sau_sregion) {

            return 0;

        }

        return cpu->env.sau.rlar[region];

    }

    case 0xde4: /* SFSR */

        if (!arm_feature(&cpu->env, ARM_FEATURE_V8)) {

            goto bad_offset;

        }

        if (!attrs.secure) {

            return 0;

        }

        return cpu->env.v7m.sfsr;

    case 0xde8: /* SFAR */

        if (!arm_feature(&cpu->env, ARM_FEATURE_V8)) {

            goto bad_offset;

        }

        if (!attrs.secure) {

            return 0;

        }

        return cpu->env.v7m.sfar;

    default:

    bad_offset:

        qemu_log_mask(LOG_GUEST_ERROR, "NVIC: Bad read offset 0x%x\n", offset);

        return 0;

    }

}
