static void nvic_writel(NVICState *s, uint32_t offset, uint32_t value,

                        MemTxAttrs attrs)

{

    ARMCPU *cpu = s->cpu;



    switch (offset) {

    case 0x380 ... 0x3bf: /* NVIC_ITNS<n> */

    {

        int startvec = 32 * (offset - 0x380) + NVIC_FIRST_IRQ;

        int i;



        if (!arm_feature(&cpu->env, ARM_FEATURE_V8)) {

            goto bad_offset;

        }

        if (!attrs.secure) {

            break;

        }

        for (i = 0; i < 32 && startvec + i < s->num_irq; i++) {

            s->itns[startvec + i] = (value >> i) & 1;

        }

        nvic_irq_update(s);

        break;

    }

    case 0xd04: /* Interrupt Control State (ICSR) */

        if (cpu->env.v7m.aircr & R_V7M_AIRCR_BFHFNMINS_MASK) {

            if (value & (1 << 31)) {

                armv7m_nvic_set_pending(s, ARMV7M_EXCP_NMI, false);

            } else if (value & (1 << 30) &&

                       arm_feature(&cpu->env, ARM_FEATURE_V8)) {

                /* PENDNMICLR didn't exist in v7M */

                armv7m_nvic_clear_pending(s, ARMV7M_EXCP_NMI, false);

            }

        }

        if (value & (1 << 28)) {

            armv7m_nvic_set_pending(s, ARMV7M_EXCP_PENDSV, attrs.secure);

        } else if (value & (1 << 27)) {

            armv7m_nvic_clear_pending(s, ARMV7M_EXCP_PENDSV, attrs.secure);

        }

        if (value & (1 << 26)) {

            armv7m_nvic_set_pending(s, ARMV7M_EXCP_SYSTICK, attrs.secure);

        } else if (value & (1 << 25)) {

            armv7m_nvic_clear_pending(s, ARMV7M_EXCP_SYSTICK, attrs.secure);

        }

        break;

    case 0xd08: /* Vector Table Offset.  */

        cpu->env.v7m.vecbase[attrs.secure] = value & 0xffffff80;

        break;

    case 0xd0c: /* Application Interrupt/Reset Control (AIRCR) */

        if ((value >> R_V7M_AIRCR_VECTKEY_SHIFT) == 0x05fa) {

            if (value & R_V7M_AIRCR_SYSRESETREQ_MASK) {

                if (attrs.secure ||

                    !(cpu->env.v7m.aircr & R_V7M_AIRCR_SYSRESETREQS_MASK)) {

                    qemu_irq_pulse(s->sysresetreq);

                }

            }

            if (value & R_V7M_AIRCR_VECTCLRACTIVE_MASK) {

                qemu_log_mask(LOG_GUEST_ERROR,

                              "Setting VECTCLRACTIVE when not in DEBUG mode "

                              "is UNPREDICTABLE\n");

            }

            if (value & R_V7M_AIRCR_VECTRESET_MASK) {

                /* NB: this bit is RES0 in v8M */

                qemu_log_mask(LOG_GUEST_ERROR,

                              "Setting VECTRESET when not in DEBUG mode "

                              "is UNPREDICTABLE\n");

            }

            s->prigroup[attrs.secure] = extract32(value,

                                                  R_V7M_AIRCR_PRIGROUP_SHIFT,

                                                  R_V7M_AIRCR_PRIGROUP_LENGTH);

            if (attrs.secure) {

                /* These bits are only writable by secure */

                cpu->env.v7m.aircr = value &

                    (R_V7M_AIRCR_SYSRESETREQS_MASK |

                     R_V7M_AIRCR_BFHFNMINS_MASK |

                     R_V7M_AIRCR_PRIS_MASK);

                /* BFHFNMINS changes the priority of Secure HardFault, and

                 * allows a pending Non-secure HardFault to preempt (which

                 * we implement by marking it enabled).

                 */

                if (cpu->env.v7m.aircr & R_V7M_AIRCR_BFHFNMINS_MASK) {

                    s->sec_vectors[ARMV7M_EXCP_HARD].prio = -3;

                    s->vectors[ARMV7M_EXCP_HARD].enabled = 1;

                } else {

                    s->sec_vectors[ARMV7M_EXCP_HARD].prio = -1;

                    s->vectors[ARMV7M_EXCP_HARD].enabled = 0;

                }

            }

            nvic_irq_update(s);

        }

        break;

    case 0xd10: /* System Control.  */

        /* TODO: Implement control registers.  */

        qemu_log_mask(LOG_UNIMP, "NVIC: SCR unimplemented\n");

        break;

    case 0xd14: /* Configuration Control.  */

        /* Enforce RAZ/WI on reserved and must-RAZ/WI bits */

        value &= (R_V7M_CCR_STKALIGN_MASK |

                  R_V7M_CCR_BFHFNMIGN_MASK |

                  R_V7M_CCR_DIV_0_TRP_MASK |

                  R_V7M_CCR_UNALIGN_TRP_MASK |

                  R_V7M_CCR_USERSETMPEND_MASK |

                  R_V7M_CCR_NONBASETHRDENA_MASK);



        if (arm_feature(&cpu->env, ARM_FEATURE_V8)) {

            /* v8M makes NONBASETHRDENA and STKALIGN be RES1 */

            value |= R_V7M_CCR_NONBASETHRDENA_MASK

                | R_V7M_CCR_STKALIGN_MASK;

        }

        if (attrs.secure) {

            /* the BFHFNMIGN bit is not banked; keep that in the NS copy */

            cpu->env.v7m.ccr[M_REG_NS] =

                (cpu->env.v7m.ccr[M_REG_NS] & ~R_V7M_CCR_BFHFNMIGN_MASK)

                | (value & R_V7M_CCR_BFHFNMIGN_MASK);

            value &= ~R_V7M_CCR_BFHFNMIGN_MASK;

        }



        cpu->env.v7m.ccr[attrs.secure] = value;

        break;

    case 0xd24: /* System Handler Control and State (SHCSR) */

        if (attrs.secure) {

            s->sec_vectors[ARMV7M_EXCP_MEM].active = (value & (1 << 0)) != 0;

            /* Secure HardFault active bit cannot be written */

            s->sec_vectors[ARMV7M_EXCP_USAGE].active = (value & (1 << 3)) != 0;

            s->sec_vectors[ARMV7M_EXCP_SVC].active = (value & (1 << 7)) != 0;

            s->sec_vectors[ARMV7M_EXCP_PENDSV].active =

                (value & (1 << 10)) != 0;

            s->sec_vectors[ARMV7M_EXCP_SYSTICK].active =

                (value & (1 << 11)) != 0;

            s->sec_vectors[ARMV7M_EXCP_USAGE].pending =

                (value & (1 << 12)) != 0;

            s->sec_vectors[ARMV7M_EXCP_MEM].pending = (value & (1 << 13)) != 0;

            s->sec_vectors[ARMV7M_EXCP_SVC].pending = (value & (1 << 15)) != 0;

            s->sec_vectors[ARMV7M_EXCP_MEM].enabled = (value & (1 << 16)) != 0;

            s->sec_vectors[ARMV7M_EXCP_BUS].enabled = (value & (1 << 17)) != 0;

            s->sec_vectors[ARMV7M_EXCP_USAGE].enabled =

                (value & (1 << 18)) != 0;

            s->sec_vectors[ARMV7M_EXCP_HARD].pending = (value & (1 << 21)) != 0;

            /* SecureFault not banked, but RAZ/WI to NS */

            s->vectors[ARMV7M_EXCP_SECURE].active = (value & (1 << 4)) != 0;

            s->vectors[ARMV7M_EXCP_SECURE].enabled = (value & (1 << 19)) != 0;

            s->vectors[ARMV7M_EXCP_SECURE].pending = (value & (1 << 20)) != 0;

        } else {

            s->vectors[ARMV7M_EXCP_MEM].active = (value & (1 << 0)) != 0;

            if (arm_feature(&cpu->env, ARM_FEATURE_V8)) {

                /* HARDFAULTPENDED is not present in v7M */

                s->vectors[ARMV7M_EXCP_HARD].pending = (value & (1 << 21)) != 0;

            }

            s->vectors[ARMV7M_EXCP_USAGE].active = (value & (1 << 3)) != 0;

            s->vectors[ARMV7M_EXCP_SVC].active = (value & (1 << 7)) != 0;

            s->vectors[ARMV7M_EXCP_PENDSV].active = (value & (1 << 10)) != 0;

            s->vectors[ARMV7M_EXCP_SYSTICK].active = (value & (1 << 11)) != 0;

            s->vectors[ARMV7M_EXCP_USAGE].pending = (value & (1 << 12)) != 0;

            s->vectors[ARMV7M_EXCP_MEM].pending = (value & (1 << 13)) != 0;

            s->vectors[ARMV7M_EXCP_SVC].pending = (value & (1 << 15)) != 0;

            s->vectors[ARMV7M_EXCP_MEM].enabled = (value & (1 << 16)) != 0;

            s->vectors[ARMV7M_EXCP_USAGE].enabled = (value & (1 << 18)) != 0;

        }

        if (attrs.secure || (cpu->env.v7m.aircr & R_V7M_AIRCR_BFHFNMINS_MASK)) {

            s->vectors[ARMV7M_EXCP_BUS].active = (value & (1 << 1)) != 0;

            s->vectors[ARMV7M_EXCP_BUS].pending = (value & (1 << 14)) != 0;

            s->vectors[ARMV7M_EXCP_BUS].enabled = (value & (1 << 17)) != 0;

        }

        /* NMIACT can only be written if the write is of a zero, with

         * BFHFNMINS 1, and by the CPU in secure state via the NS alias.

         */

        if (!attrs.secure && cpu->env.v7m.secure &&

            (cpu->env.v7m.aircr & R_V7M_AIRCR_BFHFNMINS_MASK) &&

            (value & (1 << 5)) == 0) {

            s->vectors[ARMV7M_EXCP_NMI].active = 0;

        }

        /* HARDFAULTACT can only be written if the write is of a zero

         * to the non-secure HardFault state by the CPU in secure state.

         * The only case where we can be targeting the non-secure HF state

         * when in secure state is if this is a write via the NS alias

         * and BFHFNMINS is 1.

         */

        if (!attrs.secure && cpu->env.v7m.secure &&

            (cpu->env.v7m.aircr & R_V7M_AIRCR_BFHFNMINS_MASK) &&

            (value & (1 << 2)) == 0) {

            s->vectors[ARMV7M_EXCP_HARD].active = 0;

        }



        /* TODO: this is RAZ/WI from NS if DEMCR.SDME is set */

        s->vectors[ARMV7M_EXCP_DEBUG].active = (value & (1 << 8)) != 0;

        nvic_irq_update(s);

        break;

    case 0xd28: /* Configurable Fault Status.  */

        cpu->env.v7m.cfsr[attrs.secure] &= ~value; /* W1C */

        if (attrs.secure) {

            /* The BFSR bits [15:8] are shared between security states

             * and we store them in the NS copy.

             */

            cpu->env.v7m.cfsr[M_REG_NS] &= ~(value & R_V7M_CFSR_BFSR_MASK);

        }

        break;

    case 0xd2c: /* Hard Fault Status.  */

        cpu->env.v7m.hfsr &= ~value; /* W1C */

        break;

    case 0xd30: /* Debug Fault Status.  */

        cpu->env.v7m.dfsr &= ~value; /* W1C */

        break;

    case 0xd34: /* Mem Manage Address.  */

        cpu->env.v7m.mmfar[attrs.secure] = value;

        return;

    case 0xd38: /* Bus Fault Address.  */

        cpu->env.v7m.bfar = value;

        return;

    case 0xd3c: /* Aux Fault Status.  */

        qemu_log_mask(LOG_UNIMP,

                      "NVIC: Aux fault status registers unimplemented\n");

        break;

    case 0xd90: /* MPU_TYPE */

        return; /* RO */

    case 0xd94: /* MPU_CTRL */

        if ((value &

             (R_V7M_MPU_CTRL_HFNMIENA_MASK | R_V7M_MPU_CTRL_ENABLE_MASK))

            == R_V7M_MPU_CTRL_HFNMIENA_MASK) {

            qemu_log_mask(LOG_GUEST_ERROR, "MPU_CTRL: HFNMIENA and !ENABLE is "

                          "UNPREDICTABLE\n");

        }

        cpu->env.v7m.mpu_ctrl[attrs.secure]

            = value & (R_V7M_MPU_CTRL_ENABLE_MASK |

                       R_V7M_MPU_CTRL_HFNMIENA_MASK |

                       R_V7M_MPU_CTRL_PRIVDEFENA_MASK);

        tlb_flush(CPU(cpu));

        break;

    case 0xd98: /* MPU_RNR */

        if (value >= cpu->pmsav7_dregion) {

            qemu_log_mask(LOG_GUEST_ERROR, "MPU region out of range %"

                          PRIu32 "/%" PRIu32 "\n",

                          value, cpu->pmsav7_dregion);

        } else {

            cpu->env.pmsav7.rnr[attrs.secure] = value;

        }

        break;

    case 0xd9c: /* MPU_RBAR */

    case 0xda4: /* MPU_RBAR_A1 */

    case 0xdac: /* MPU_RBAR_A2 */

    case 0xdb4: /* MPU_RBAR_A3 */

    {

        int region;



        if (arm_feature(&cpu->env, ARM_FEATURE_V8)) {

            /* PMSAv8M handling of the aliases is different from v7M:

             * aliases A1, A2, A3 override the low two bits of the region

             * number in MPU_RNR, and there is no 'region' field in the

             * RBAR register.

             */

            int aliasno = (offset - 0xd9c) / 8; /* 0..3 */



            region = cpu->env.pmsav7.rnr[attrs.secure];

            if (aliasno) {

                region = deposit32(region, 0, 2, aliasno);

            }

            if (region >= cpu->pmsav7_dregion) {

                return;

            }

            cpu->env.pmsav8.rbar[attrs.secure][region] = value;

            tlb_flush(CPU(cpu));

            return;

        }



        if (value & (1 << 4)) {

            /* VALID bit means use the region number specified in this

             * value and also update MPU_RNR.REGION with that value.

             */

            region = extract32(value, 0, 4);

            if (region >= cpu->pmsav7_dregion) {

                qemu_log_mask(LOG_GUEST_ERROR,

                              "MPU region out of range %u/%" PRIu32 "\n",

                              region, cpu->pmsav7_dregion);

                return;

            }

            cpu->env.pmsav7.rnr[attrs.secure] = region;

        } else {

            region = cpu->env.pmsav7.rnr[attrs.secure];

        }



        if (region >= cpu->pmsav7_dregion) {

            return;

        }



        cpu->env.pmsav7.drbar[region] = value & ~0x1f;

        tlb_flush(CPU(cpu));

        break;

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

            int aliasno = (offset - 0xd9c) / 8; /* 0..3 */



            region = cpu->env.pmsav7.rnr[attrs.secure];

            if (aliasno) {

                region = deposit32(region, 0, 2, aliasno);

            }

            if (region >= cpu->pmsav7_dregion) {

                return;

            }

            cpu->env.pmsav8.rlar[attrs.secure][region] = value;

            tlb_flush(CPU(cpu));

            return;

        }



        if (region >= cpu->pmsav7_dregion) {

            return;

        }



        cpu->env.pmsav7.drsr[region] = value & 0xff3f;

        cpu->env.pmsav7.dracr[region] = (value >> 16) & 0x173f;

        tlb_flush(CPU(cpu));

        break;

    }

    case 0xdc0: /* MPU_MAIR0 */

        if (!arm_feature(&cpu->env, ARM_FEATURE_V8)) {

            goto bad_offset;

        }

        if (cpu->pmsav7_dregion) {

            /* Register is RES0 if no MPU regions are implemented */

            cpu->env.pmsav8.mair0[attrs.secure] = value;

        }

        /* We don't need to do anything else because memory attributes

         * only affect cacheability, and we don't implement caching.

         */

        break;

    case 0xdc4: /* MPU_MAIR1 */

        if (!arm_feature(&cpu->env, ARM_FEATURE_V8)) {

            goto bad_offset;

        }

        if (cpu->pmsav7_dregion) {

            /* Register is RES0 if no MPU regions are implemented */

            cpu->env.pmsav8.mair1[attrs.secure] = value;

        }

        /* We don't need to do anything else because memory attributes

         * only affect cacheability, and we don't implement caching.

         */

        break;

    case 0xdd0: /* SAU_CTRL */

        if (!arm_feature(&cpu->env, ARM_FEATURE_V8)) {

            goto bad_offset;

        }

        if (!attrs.secure) {

            return;

        }

        cpu->env.sau.ctrl = value & 3;

        break;

    case 0xdd4: /* SAU_TYPE */

        if (!arm_feature(&cpu->env, ARM_FEATURE_V8)) {

            goto bad_offset;

        }

        break;

    case 0xdd8: /* SAU_RNR */

        if (!arm_feature(&cpu->env, ARM_FEATURE_V8)) {

            goto bad_offset;

        }

        if (!attrs.secure) {

            return;

        }

        if (value >= cpu->sau_sregion) {

            qemu_log_mask(LOG_GUEST_ERROR, "SAU region out of range %"

                          PRIu32 "/%" PRIu32 "\n",

                          value, cpu->sau_sregion);

        } else {

            cpu->env.sau.rnr = value;

        }

        break;

    case 0xddc: /* SAU_RBAR */

    {

        int region = cpu->env.sau.rnr;



        if (!arm_feature(&cpu->env, ARM_FEATURE_V8)) {

            goto bad_offset;

        }

        if (!attrs.secure) {

            return;

        }

        if (region >= cpu->sau_sregion) {

            return;

        }

        cpu->env.sau.rbar[region] = value & ~0x1f;

        tlb_flush(CPU(cpu));

        break;

    }

    case 0xde0: /* SAU_RLAR */

    {

        int region = cpu->env.sau.rnr;



        if (!arm_feature(&cpu->env, ARM_FEATURE_V8)) {

            goto bad_offset;

        }

        if (!attrs.secure) {

            return;

        }

        if (region >= cpu->sau_sregion) {

            return;

        }

        cpu->env.sau.rlar[region] = value & ~0x1c;

        tlb_flush(CPU(cpu));

        break;

    }

    case 0xde4: /* SFSR */

        if (!arm_feature(&cpu->env, ARM_FEATURE_V8)) {

            goto bad_offset;

        }

        if (!attrs.secure) {

            return;

        }

        cpu->env.v7m.sfsr &= ~value; /* W1C */

        break;

    case 0xde8: /* SFAR */

        if (!arm_feature(&cpu->env, ARM_FEATURE_V8)) {

            goto bad_offset;

        }

        if (!attrs.secure) {

            return;

        }

        cpu->env.v7m.sfsr = value;

        break;

    case 0xf00: /* Software Triggered Interrupt Register */

    {

        int excnum = (value & 0x1ff) + NVIC_FIRST_IRQ;

        if (excnum < s->num_irq) {

            armv7m_nvic_set_pending(s, excnum, false);

        }

        break;

    }

    default:

    bad_offset:

        qemu_log_mask(LOG_GUEST_ERROR,

                      "NVIC: Bad write offset 0x%x\n", offset);

    }

}
