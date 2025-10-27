static void nvic_writel(NVICState *s, uint32_t offset, uint32_t value)

{

    ARMCPU *cpu = s->cpu;



    switch (offset) {

    case 0xd04: /* Interrupt Control State.  */

        if (value & (1 << 31)) {

            armv7m_nvic_set_pending(s, ARMV7M_EXCP_NMI);

        }

        if (value & (1 << 28)) {

            armv7m_nvic_set_pending(s, ARMV7M_EXCP_PENDSV);

        } else if (value & (1 << 27)) {

            armv7m_nvic_clear_pending(s, ARMV7M_EXCP_PENDSV);

        }

        if (value & (1 << 26)) {

            armv7m_nvic_set_pending(s, ARMV7M_EXCP_SYSTICK);

        } else if (value & (1 << 25)) {

            armv7m_nvic_clear_pending(s, ARMV7M_EXCP_SYSTICK);

        }

        break;

    case 0xd08: /* Vector Table Offset.  */

        cpu->env.v7m.vecbase = value & 0xffffff80;

        break;

    case 0xd0c: /* Application Interrupt/Reset Control.  */

        if ((value >> 16) == 0x05fa) {

            if (value & 4) {

                qemu_irq_pulse(s->sysresetreq);

            }

            if (value & 2) {

                qemu_log_mask(LOG_GUEST_ERROR,

                              "Setting VECTCLRACTIVE when not in DEBUG mode "

                              "is UNPREDICTABLE\n");

            }

            if (value & 1) {

                qemu_log_mask(LOG_GUEST_ERROR,

                              "Setting VECTRESET when not in DEBUG mode "

                              "is UNPREDICTABLE\n");

            }

            s->prigroup = extract32(value, 8, 3);

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



        cpu->env.v7m.ccr = value;

        break;

    case 0xd24: /* System Handler Control.  */

        s->vectors[ARMV7M_EXCP_MEM].active = (value & (1 << 0)) != 0;

        s->vectors[ARMV7M_EXCP_BUS].active = (value & (1 << 1)) != 0;

        s->vectors[ARMV7M_EXCP_USAGE].active = (value & (1 << 3)) != 0;

        s->vectors[ARMV7M_EXCP_SVC].active = (value & (1 << 7)) != 0;

        s->vectors[ARMV7M_EXCP_DEBUG].active = (value & (1 << 8)) != 0;

        s->vectors[ARMV7M_EXCP_PENDSV].active = (value & (1 << 10)) != 0;

        s->vectors[ARMV7M_EXCP_SYSTICK].active = (value & (1 << 11)) != 0;

        s->vectors[ARMV7M_EXCP_USAGE].pending = (value & (1 << 12)) != 0;

        s->vectors[ARMV7M_EXCP_MEM].pending = (value & (1 << 13)) != 0;

        s->vectors[ARMV7M_EXCP_BUS].pending = (value & (1 << 14)) != 0;

        s->vectors[ARMV7M_EXCP_SVC].pending = (value & (1 << 15)) != 0;

        s->vectors[ARMV7M_EXCP_MEM].enabled = (value & (1 << 16)) != 0;

        s->vectors[ARMV7M_EXCP_BUS].enabled = (value & (1 << 17)) != 0;

        s->vectors[ARMV7M_EXCP_USAGE].enabled = (value & (1 << 18)) != 0;

        nvic_irq_update(s);

        break;

    case 0xd28: /* Configurable Fault Status.  */

        cpu->env.v7m.cfsr &= ~value; /* W1C */

        break;

    case 0xd2c: /* Hard Fault Status.  */

        cpu->env.v7m.hfsr &= ~value; /* W1C */

        break;

    case 0xd30: /* Debug Fault Status.  */

        cpu->env.v7m.dfsr &= ~value; /* W1C */

        break;

    case 0xd34: /* Mem Manage Address.  */

        cpu->env.v7m.mmfar = value;

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

        cpu->env.v7m.mpu_ctrl = value & (R_V7M_MPU_CTRL_ENABLE_MASK |

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

            cpu->env.pmsav7.rnr = value;

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



            region = cpu->env.pmsav7.rnr;

            if (aliasno) {

                region = deposit32(region, 0, 2, aliasno);

            }

            if (region >= cpu->pmsav7_dregion) {

                return;

            }

            cpu->env.pmsav8.rbar[region] = value;

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

            cpu->env.pmsav7.rnr = region;

        } else {

            region = cpu->env.pmsav7.rnr;

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

        int region = cpu->env.pmsav7.rnr;



        if (arm_feature(&cpu->env, ARM_FEATURE_V8)) {

            /* PMSAv8M handling of the aliases is different from v7M:

             * aliases A1, A2, A3 override the low two bits of the region

             * number in MPU_RNR.

             */

            int aliasno = (offset - 0xd9c) / 8; /* 0..3 */



            region = cpu->env.pmsav7.rnr;

            if (aliasno) {

                region = deposit32(region, 0, 2, aliasno);

            }

            if (region >= cpu->pmsav7_dregion) {

                return;

            }

            cpu->env.pmsav8.rlar[region] = value;

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

            cpu->env.pmsav8.mair0 = value;

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

            cpu->env.pmsav8.mair1 = value;

        }

        /* We don't need to do anything else because memory attributes

         * only affect cacheability, and we don't implement caching.

         */

        break;

    case 0xf00: /* Software Triggered Interrupt Register */

    {

        int excnum = (value & 0x1ff) + NVIC_FIRST_IRQ;

        if (excnum < s->num_irq) {

            armv7m_nvic_set_pending(s, excnum);

        }

        break;

    }

    default:

    bad_offset:

        qemu_log_mask(LOG_GUEST_ERROR,

                      "NVIC: Bad write offset 0x%x\n", offset);

    }

}
