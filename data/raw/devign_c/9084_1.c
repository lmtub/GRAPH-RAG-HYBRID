void register_cp_regs_for_features(ARMCPU *cpu)

{

    /* Register all the coprocessor registers based on feature bits */

    CPUARMState *env = &cpu->env;

    if (arm_feature(env, ARM_FEATURE_M)) {

        /* M profile has no coprocessor registers */

        return;

    }



    define_arm_cp_regs(cpu, cp_reginfo);

    if (!arm_feature(env, ARM_FEATURE_V8)) {

        /* Must go early as it is full of wildcards that may be

         * overridden by later definitions.

         */

        define_arm_cp_regs(cpu, not_v8_cp_reginfo);

    }



    if (arm_feature(env, ARM_FEATURE_V6)) {

        /* The ID registers all have impdef reset values */

        ARMCPRegInfo v6_idregs[] = {

            { .name = "ID_PFR0", .state = ARM_CP_STATE_BOTH,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 1, .opc2 = 0,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->id_pfr0 },

            { .name = "ID_PFR1", .state = ARM_CP_STATE_BOTH,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 1, .opc2 = 1,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->id_pfr1 },

            { .name = "ID_DFR0", .state = ARM_CP_STATE_BOTH,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 1, .opc2 = 2,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->id_dfr0 },

            { .name = "ID_AFR0", .state = ARM_CP_STATE_BOTH,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 1, .opc2 = 3,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->id_afr0 },

            { .name = "ID_MMFR0", .state = ARM_CP_STATE_BOTH,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 1, .opc2 = 4,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->id_mmfr0 },

            { .name = "ID_MMFR1", .state = ARM_CP_STATE_BOTH,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 1, .opc2 = 5,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->id_mmfr1 },

            { .name = "ID_MMFR2", .state = ARM_CP_STATE_BOTH,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 1, .opc2 = 6,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->id_mmfr2 },

            { .name = "ID_MMFR3", .state = ARM_CP_STATE_BOTH,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 1, .opc2 = 7,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->id_mmfr3 },

            { .name = "ID_ISAR0", .state = ARM_CP_STATE_BOTH,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 2, .opc2 = 0,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->id_isar0 },

            { .name = "ID_ISAR1", .state = ARM_CP_STATE_BOTH,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 2, .opc2 = 1,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->id_isar1 },

            { .name = "ID_ISAR2", .state = ARM_CP_STATE_BOTH,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 2, .opc2 = 2,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->id_isar2 },

            { .name = "ID_ISAR3", .state = ARM_CP_STATE_BOTH,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 2, .opc2 = 3,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->id_isar3 },

            { .name = "ID_ISAR4", .state = ARM_CP_STATE_BOTH,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 2, .opc2 = 4,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->id_isar4 },

            { .name = "ID_ISAR5", .state = ARM_CP_STATE_BOTH,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 2, .opc2 = 5,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->id_isar5 },

            { .name = "ID_MMFR4", .state = ARM_CP_STATE_BOTH,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 2, .opc2 = 6,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->id_mmfr4 },

            /* 7 is as yet unallocated and must RAZ */

            { .name = "ID_ISAR7_RESERVED", .state = ARM_CP_STATE_BOTH,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 2, .opc2 = 7,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = 0 },

            REGINFO_SENTINEL

        };

        define_arm_cp_regs(cpu, v6_idregs);

        define_arm_cp_regs(cpu, v6_cp_reginfo);

    } else {

        define_arm_cp_regs(cpu, not_v6_cp_reginfo);

    }

    if (arm_feature(env, ARM_FEATURE_V6K)) {

        define_arm_cp_regs(cpu, v6k_cp_reginfo);

    }

    if (arm_feature(env, ARM_FEATURE_V7MP) &&

        !arm_feature(env, ARM_FEATURE_PMSA)) {

        define_arm_cp_regs(cpu, v7mp_cp_reginfo);

    }

    if (arm_feature(env, ARM_FEATURE_V7)) {

        /* v7 performance monitor control register: same implementor

         * field as main ID register, and we implement only the cycle

         * count register.

         */

#ifndef CONFIG_USER_ONLY

        ARMCPRegInfo pmcr = {

            .name = "PMCR", .cp = 15, .crn = 9, .crm = 12, .opc1 = 0, .opc2 = 0,

            .access = PL0_RW,

            .type = ARM_CP_IO | ARM_CP_ALIAS,

            .fieldoffset = offsetoflow32(CPUARMState, cp15.c9_pmcr),

            .accessfn = pmreg_access, .writefn = pmcr_write,

            .raw_writefn = raw_write,

        };

        ARMCPRegInfo pmcr64 = {

            .name = "PMCR_EL0", .state = ARM_CP_STATE_AA64,

            .opc0 = 3, .opc1 = 3, .crn = 9, .crm = 12, .opc2 = 0,

            .access = PL0_RW, .accessfn = pmreg_access,

            .type = ARM_CP_IO,

            .fieldoffset = offsetof(CPUARMState, cp15.c9_pmcr),

            .resetvalue = cpu->midr & 0xff000000,

            .writefn = pmcr_write, .raw_writefn = raw_write,

        };

        define_one_arm_cp_reg(cpu, &pmcr);

        define_one_arm_cp_reg(cpu, &pmcr64);

#endif

        ARMCPRegInfo clidr = {

            .name = "CLIDR", .state = ARM_CP_STATE_BOTH,

            .opc0 = 3, .crn = 0, .crm = 0, .opc1 = 1, .opc2 = 1,

            .access = PL1_R, .type = ARM_CP_CONST, .resetvalue = cpu->clidr

        };

        define_one_arm_cp_reg(cpu, &clidr);

        define_arm_cp_regs(cpu, v7_cp_reginfo);

        define_debug_regs(cpu);

    } else {

        define_arm_cp_regs(cpu, not_v7_cp_reginfo);

    }

    if (arm_feature(env, ARM_FEATURE_V8)) {

        /* AArch64 ID registers, which all have impdef reset values.

         * Note that within the ID register ranges the unused slots

         * must all RAZ, not UNDEF; future architecture versions may

         * define new registers here.

         */

        ARMCPRegInfo v8_idregs[] = {

            { .name = "ID_AA64PFR0_EL1", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 4, .opc2 = 0,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->id_aa64pfr0 },

            { .name = "ID_AA64PFR1_EL1", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 4, .opc2 = 1,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->id_aa64pfr1},

            { .name = "ID_AA64PFR2_EL1_RESERVED", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 4, .opc2 = 2,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = 0 },

            { .name = "ID_AA64PFR3_EL1_RESERVED", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 4, .opc2 = 3,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = 0 },

            { .name = "ID_AA64PFR4_EL1_RESERVED", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 4, .opc2 = 4,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = 0 },

            { .name = "ID_AA64PFR5_EL1_RESERVED", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 4, .opc2 = 5,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = 0 },

            { .name = "ID_AA64PFR6_EL1_RESERVED", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 4, .opc2 = 6,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = 0 },

            { .name = "ID_AA64PFR7_EL1_RESERVED", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 4, .opc2 = 7,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = 0 },

            { .name = "ID_AA64DFR0_EL1", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 5, .opc2 = 0,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->id_aa64dfr0 },

            { .name = "ID_AA64DFR1_EL1", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 5, .opc2 = 1,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->id_aa64dfr1 },

            { .name = "ID_AA64DFR2_EL1_RESERVED", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 5, .opc2 = 2,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = 0 },

            { .name = "ID_AA64DFR3_EL1_RESERVED", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 5, .opc2 = 3,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = 0 },

            { .name = "ID_AA64AFR0_EL1", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 5, .opc2 = 4,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->id_aa64afr0 },

            { .name = "ID_AA64AFR1_EL1", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 5, .opc2 = 5,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->id_aa64afr1 },

            { .name = "ID_AA64AFR2_EL1_RESERVED", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 5, .opc2 = 6,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = 0 },

            { .name = "ID_AA64AFR3_EL1_RESERVED", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 5, .opc2 = 7,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = 0 },

            { .name = "ID_AA64ISAR0_EL1", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 6, .opc2 = 0,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->id_aa64isar0 },

            { .name = "ID_AA64ISAR1_EL1", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 6, .opc2 = 1,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->id_aa64isar1 },

            { .name = "ID_AA64ISAR2_EL1_RESERVED", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 6, .opc2 = 2,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = 0 },

            { .name = "ID_AA64ISAR3_EL1_RESERVED", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 6, .opc2 = 3,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = 0 },

            { .name = "ID_AA64ISAR4_EL1_RESERVED", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 6, .opc2 = 4,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = 0 },

            { .name = "ID_AA64ISAR5_EL1_RESERVED", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 6, .opc2 = 5,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = 0 },

            { .name = "ID_AA64ISAR6_EL1_RESERVED", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 6, .opc2 = 6,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = 0 },

            { .name = "ID_AA64ISAR7_EL1_RESERVED", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 6, .opc2 = 7,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = 0 },

            { .name = "ID_AA64MMFR0_EL1", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 7, .opc2 = 0,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->id_aa64mmfr0 },

            { .name = "ID_AA64MMFR1_EL1", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 7, .opc2 = 1,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->id_aa64mmfr1 },

            { .name = "ID_AA64MMFR2_EL1_RESERVED", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 7, .opc2 = 2,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = 0 },

            { .name = "ID_AA64MMFR3_EL1_RESERVED", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 7, .opc2 = 3,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = 0 },

            { .name = "ID_AA64MMFR4_EL1_RESERVED", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 7, .opc2 = 4,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = 0 },

            { .name = "ID_AA64MMFR5_EL1_RESERVED", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 7, .opc2 = 5,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = 0 },

            { .name = "ID_AA64MMFR6_EL1_RESERVED", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 7, .opc2 = 6,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = 0 },

            { .name = "ID_AA64MMFR7_EL1_RESERVED", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 7, .opc2 = 7,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = 0 },

            { .name = "MVFR0_EL1", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 3, .opc2 = 0,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->mvfr0 },

            { .name = "MVFR1_EL1", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 3, .opc2 = 1,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->mvfr1 },

            { .name = "MVFR2_EL1", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 3, .opc2 = 2,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->mvfr2 },

            { .name = "MVFR3_EL1_RESERVED", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 3, .opc2 = 3,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = 0 },

            { .name = "MVFR4_EL1_RESERVED", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 3, .opc2 = 4,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = 0 },

            { .name = "MVFR5_EL1_RESERVED", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 3, .opc2 = 5,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = 0 },

            { .name = "MVFR6_EL1_RESERVED", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 3, .opc2 = 6,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = 0 },

            { .name = "MVFR7_EL1_RESERVED", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 3, .opc2 = 7,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = 0 },

            { .name = "PMCEID0", .state = ARM_CP_STATE_AA32,

              .cp = 15, .opc1 = 0, .crn = 9, .crm = 12, .opc2 = 6,

              .access = PL0_R, .accessfn = pmreg_access, .type = ARM_CP_CONST,

              .resetvalue = cpu->pmceid0 },

            { .name = "PMCEID0_EL0", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 3, .crn = 9, .crm = 12, .opc2 = 6,

              .access = PL0_R, .accessfn = pmreg_access, .type = ARM_CP_CONST,

              .resetvalue = cpu->pmceid0 },

            { .name = "PMCEID1", .state = ARM_CP_STATE_AA32,

              .cp = 15, .opc1 = 0, .crn = 9, .crm = 12, .opc2 = 7,

              .access = PL0_R, .accessfn = pmreg_access, .type = ARM_CP_CONST,

              .resetvalue = cpu->pmceid1 },

            { .name = "PMCEID1_EL0", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 3, .crn = 9, .crm = 12, .opc2 = 7,

              .access = PL0_R, .accessfn = pmreg_access, .type = ARM_CP_CONST,

              .resetvalue = cpu->pmceid1 },

            REGINFO_SENTINEL

        };

        /* RVBAR_EL1 is only implemented if EL1 is the highest EL */

        if (!arm_feature(env, ARM_FEATURE_EL3) &&

            !arm_feature(env, ARM_FEATURE_EL2)) {

            ARMCPRegInfo rvbar = {

                .name = "RVBAR_EL1", .state = ARM_CP_STATE_AA64,

                .opc0 = 3, .opc1 = 0, .crn = 12, .crm = 0, .opc2 = 1,

                .type = ARM_CP_CONST, .access = PL1_R, .resetvalue = cpu->rvbar

            };

            define_one_arm_cp_reg(cpu, &rvbar);

        }

        define_arm_cp_regs(cpu, v8_idregs);

        define_arm_cp_regs(cpu, v8_cp_reginfo);

    }

    if (arm_feature(env, ARM_FEATURE_EL2)) {

        uint64_t vmpidr_def = mpidr_read_val(env);

        ARMCPRegInfo vpidr_regs[] = {

            { .name = "VPIDR", .state = ARM_CP_STATE_AA32,

              .cp = 15, .opc1 = 4, .crn = 0, .crm = 0, .opc2 = 0,

              .access = PL2_RW, .accessfn = access_el3_aa32ns,

              .resetvalue = cpu->midr,

              .fieldoffset = offsetof(CPUARMState, cp15.vpidr_el2) },

            { .name = "VPIDR_EL2", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 4, .crn = 0, .crm = 0, .opc2 = 0,

              .access = PL2_RW, .resetvalue = cpu->midr,

              .fieldoffset = offsetof(CPUARMState, cp15.vpidr_el2) },

            { .name = "VMPIDR", .state = ARM_CP_STATE_AA32,

              .cp = 15, .opc1 = 4, .crn = 0, .crm = 0, .opc2 = 5,

              .access = PL2_RW, .accessfn = access_el3_aa32ns,

              .resetvalue = vmpidr_def,

              .fieldoffset = offsetof(CPUARMState, cp15.vmpidr_el2) },

            { .name = "VMPIDR_EL2", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 4, .crn = 0, .crm = 0, .opc2 = 5,

              .access = PL2_RW,

              .resetvalue = vmpidr_def,

              .fieldoffset = offsetof(CPUARMState, cp15.vmpidr_el2) },

            REGINFO_SENTINEL

        };

        define_arm_cp_regs(cpu, vpidr_regs);

        define_arm_cp_regs(cpu, el2_cp_reginfo);

        /* RVBAR_EL2 is only implemented if EL2 is the highest EL */

        if (!arm_feature(env, ARM_FEATURE_EL3)) {

            ARMCPRegInfo rvbar = {

                .name = "RVBAR_EL2", .state = ARM_CP_STATE_AA64,

                .opc0 = 3, .opc1 = 4, .crn = 12, .crm = 0, .opc2 = 1,

                .type = ARM_CP_CONST, .access = PL2_R, .resetvalue = cpu->rvbar

            };

            define_one_arm_cp_reg(cpu, &rvbar);

        }

    } else {

        /* If EL2 is missing but higher ELs are enabled, we need to

         * register the no_el2 reginfos.

         */

        if (arm_feature(env, ARM_FEATURE_EL3)) {

            /* When EL3 exists but not EL2, VPIDR and VMPIDR take the value

             * of MIDR_EL1 and MPIDR_EL1.

             */

            ARMCPRegInfo vpidr_regs[] = {

                { .name = "VPIDR_EL2", .state = ARM_CP_STATE_BOTH,

                  .opc0 = 3, .opc1 = 4, .crn = 0, .crm = 0, .opc2 = 0,

                  .access = PL2_RW, .accessfn = access_el3_aa32ns_aa64any,

                  .type = ARM_CP_CONST, .resetvalue = cpu->midr,

                  .fieldoffset = offsetof(CPUARMState, cp15.vpidr_el2) },

                { .name = "VMPIDR_EL2", .state = ARM_CP_STATE_BOTH,

                  .opc0 = 3, .opc1 = 4, .crn = 0, .crm = 0, .opc2 = 5,

                  .access = PL2_RW, .accessfn = access_el3_aa32ns_aa64any,

                  .type = ARM_CP_NO_RAW,

                  .writefn = arm_cp_write_ignore, .readfn = mpidr_read },

                REGINFO_SENTINEL

            };

            define_arm_cp_regs(cpu, vpidr_regs);

            define_arm_cp_regs(cpu, el3_no_el2_cp_reginfo);

        }

    }

    if (arm_feature(env, ARM_FEATURE_EL3)) {

        define_arm_cp_regs(cpu, el3_cp_reginfo);

        ARMCPRegInfo el3_regs[] = {

            { .name = "RVBAR_EL3", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 6, .crn = 12, .crm = 0, .opc2 = 1,

              .type = ARM_CP_CONST, .access = PL3_R, .resetvalue = cpu->rvbar },

            { .name = "SCTLR_EL3", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 6, .crn = 1, .crm = 0, .opc2 = 0,

              .access = PL3_RW,

              .raw_writefn = raw_write, .writefn = sctlr_write,

              .fieldoffset = offsetof(CPUARMState, cp15.sctlr_el[3]),

              .resetvalue = cpu->reset_sctlr },

            REGINFO_SENTINEL

        };



        define_arm_cp_regs(cpu, el3_regs);

    }

    /* The behaviour of NSACR is sufficiently various that we don't

     * try to describe it in a single reginfo:

     *  if EL3 is 64 bit, then trap to EL3 from S EL1,

     *     reads as constant 0xc00 from NS EL1 and NS EL2

     *  if EL3 is 32 bit, then RW at EL3, RO at NS EL1 and NS EL2

     *  if v7 without EL3, register doesn't exist

     *  if v8 without EL3, reads as constant 0xc00 from NS EL1 and NS EL2

     */

    if (arm_feature(env, ARM_FEATURE_EL3)) {

        if (arm_feature(env, ARM_FEATURE_AARCH64)) {

            ARMCPRegInfo nsacr = {

                .name = "NSACR", .type = ARM_CP_CONST,

                .cp = 15, .opc1 = 0, .crn = 1, .crm = 1, .opc2 = 2,

                .access = PL1_RW, .accessfn = nsacr_access,

                .resetvalue = 0xc00

            };

            define_one_arm_cp_reg(cpu, &nsacr);

        } else {

            ARMCPRegInfo nsacr = {

                .name = "NSACR",

                .cp = 15, .opc1 = 0, .crn = 1, .crm = 1, .opc2 = 2,

                .access = PL3_RW | PL1_R,

                .resetvalue = 0,

                .fieldoffset = offsetof(CPUARMState, cp15.nsacr)

            };

            define_one_arm_cp_reg(cpu, &nsacr);

        }

    } else {

        if (arm_feature(env, ARM_FEATURE_V8)) {

            ARMCPRegInfo nsacr = {

                .name = "NSACR", .type = ARM_CP_CONST,

                .cp = 15, .opc1 = 0, .crn = 1, .crm = 1, .opc2 = 2,

                .access = PL1_R,

                .resetvalue = 0xc00

            };

            define_one_arm_cp_reg(cpu, &nsacr);

        }

    }



    if (arm_feature(env, ARM_FEATURE_PMSA)) {

        if (arm_feature(env, ARM_FEATURE_V6)) {

            /* PMSAv6 not implemented */

            assert(arm_feature(env, ARM_FEATURE_V7));

            define_arm_cp_regs(cpu, vmsa_pmsa_cp_reginfo);

            define_arm_cp_regs(cpu, pmsav7_cp_reginfo);

        } else {

            define_arm_cp_regs(cpu, pmsav5_cp_reginfo);

        }

    } else {

        define_arm_cp_regs(cpu, vmsa_pmsa_cp_reginfo);

        define_arm_cp_regs(cpu, vmsa_cp_reginfo);

    }

    if (arm_feature(env, ARM_FEATURE_THUMB2EE)) {

        define_arm_cp_regs(cpu, t2ee_cp_reginfo);

    }

    if (arm_feature(env, ARM_FEATURE_GENERIC_TIMER)) {

        define_arm_cp_regs(cpu, generic_timer_cp_reginfo);

    }

    if (arm_feature(env, ARM_FEATURE_VAPA)) {

        define_arm_cp_regs(cpu, vapa_cp_reginfo);

    }

    if (arm_feature(env, ARM_FEATURE_CACHE_TEST_CLEAN)) {

        define_arm_cp_regs(cpu, cache_test_clean_cp_reginfo);

    }

    if (arm_feature(env, ARM_FEATURE_CACHE_DIRTY_REG)) {

        define_arm_cp_regs(cpu, cache_dirty_status_cp_reginfo);

    }

    if (arm_feature(env, ARM_FEATURE_CACHE_BLOCK_OPS)) {

        define_arm_cp_regs(cpu, cache_block_ops_cp_reginfo);

    }

    if (arm_feature(env, ARM_FEATURE_OMAPCP)) {

        define_arm_cp_regs(cpu, omap_cp_reginfo);

    }

    if (arm_feature(env, ARM_FEATURE_STRONGARM)) {

        define_arm_cp_regs(cpu, strongarm_cp_reginfo);

    }

    if (arm_feature(env, ARM_FEATURE_XSCALE)) {

        define_arm_cp_regs(cpu, xscale_cp_reginfo);

    }

    if (arm_feature(env, ARM_FEATURE_DUMMY_C15_REGS)) {

        define_arm_cp_regs(cpu, dummy_c15_cp_reginfo);

    }

    if (arm_feature(env, ARM_FEATURE_LPAE)) {

        define_arm_cp_regs(cpu, lpae_cp_reginfo);

    }

    /* Slightly awkwardly, the OMAP and StrongARM cores need all of

     * cp15 crn=0 to be writes-ignored, whereas for other cores they should

     * be read-only (ie write causes UNDEF exception).

     */

    {

        ARMCPRegInfo id_pre_v8_midr_cp_reginfo[] = {

            /* Pre-v8 MIDR space.

             * Note that the MIDR isn't a simple constant register because

             * of the TI925 behaviour where writes to another register can

             * cause the MIDR value to change.

             *

             * Unimplemented registers in the c15 0 0 0 space default to

             * MIDR. Define MIDR first as this entire space, then CTR, TCMTR

             * and friends override accordingly.

             */

            { .name = "MIDR",

              .cp = 15, .crn = 0, .crm = 0, .opc1 = 0, .opc2 = CP_ANY,

              .access = PL1_R, .resetvalue = cpu->midr,

              .writefn = arm_cp_write_ignore, .raw_writefn = raw_write,

              .readfn = midr_read,

              .fieldoffset = offsetof(CPUARMState, cp15.c0_cpuid),

              .type = ARM_CP_OVERRIDE },

            /* crn = 0 op1 = 0 crm = 3..7 : currently unassigned; we RAZ. */

            { .name = "DUMMY",

              .cp = 15, .crn = 0, .crm = 3, .opc1 = 0, .opc2 = CP_ANY,

              .access = PL1_R, .type = ARM_CP_CONST, .resetvalue = 0 },

            { .name = "DUMMY",

              .cp = 15, .crn = 0, .crm = 4, .opc1 = 0, .opc2 = CP_ANY,

              .access = PL1_R, .type = ARM_CP_CONST, .resetvalue = 0 },

            { .name = "DUMMY",

              .cp = 15, .crn = 0, .crm = 5, .opc1 = 0, .opc2 = CP_ANY,

              .access = PL1_R, .type = ARM_CP_CONST, .resetvalue = 0 },

            { .name = "DUMMY",

              .cp = 15, .crn = 0, .crm = 6, .opc1 = 0, .opc2 = CP_ANY,

              .access = PL1_R, .type = ARM_CP_CONST, .resetvalue = 0 },

            { .name = "DUMMY",

              .cp = 15, .crn = 0, .crm = 7, .opc1 = 0, .opc2 = CP_ANY,

              .access = PL1_R, .type = ARM_CP_CONST, .resetvalue = 0 },

            REGINFO_SENTINEL

        };

        ARMCPRegInfo id_v8_midr_cp_reginfo[] = {

            { .name = "MIDR_EL1", .state = ARM_CP_STATE_BOTH,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 0, .opc2 = 0,

              .access = PL1_R, .type = ARM_CP_NO_RAW, .resetvalue = cpu->midr,

              .fieldoffset = offsetof(CPUARMState, cp15.c0_cpuid),

              .readfn = midr_read },

            /* crn = 0 op1 = 0 crm = 0 op2 = 4,7 : AArch32 aliases of MIDR */

            { .name = "MIDR", .type = ARM_CP_ALIAS | ARM_CP_CONST,

              .cp = 15, .crn = 0, .crm = 0, .opc1 = 0, .opc2 = 4,

              .access = PL1_R, .resetvalue = cpu->midr },

            { .name = "MIDR", .type = ARM_CP_ALIAS | ARM_CP_CONST,

              .cp = 15, .crn = 0, .crm = 0, .opc1 = 0, .opc2 = 7,

              .access = PL1_R, .resetvalue = cpu->midr },

            { .name = "REVIDR_EL1", .state = ARM_CP_STATE_BOTH,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 0, .opc2 = 6,

              .access = PL1_R, .type = ARM_CP_CONST, .resetvalue = cpu->revidr },

            REGINFO_SENTINEL

        };

        ARMCPRegInfo id_cp_reginfo[] = {

            /* These are common to v8 and pre-v8 */

            { .name = "CTR",

              .cp = 15, .crn = 0, .crm = 0, .opc1 = 0, .opc2 = 1,

              .access = PL1_R, .type = ARM_CP_CONST, .resetvalue = cpu->ctr },

            { .name = "CTR_EL0", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 3, .opc2 = 1, .crn = 0, .crm = 0,

              .access = PL0_R, .accessfn = ctr_el0_access,

              .type = ARM_CP_CONST, .resetvalue = cpu->ctr },

            /* TCMTR and TLBTR exist in v8 but have no 64-bit versions */

            { .name = "TCMTR",

              .cp = 15, .crn = 0, .crm = 0, .opc1 = 0, .opc2 = 2,

              .access = PL1_R, .type = ARM_CP_CONST, .resetvalue = 0 },

            REGINFO_SENTINEL

        };

        /* TLBTR is specific to VMSA */

        ARMCPRegInfo id_tlbtr_reginfo = {

              .name = "TLBTR",

              .cp = 15, .crn = 0, .crm = 0, .opc1 = 0, .opc2 = 3,

              .access = PL1_R, .type = ARM_CP_CONST, .resetvalue = 0,

        };

        /* MPUIR is specific to PMSA V6+ */

        ARMCPRegInfo id_mpuir_reginfo = {

              .name = "MPUIR",

              .cp = 15, .crn = 0, .crm = 0, .opc1 = 0, .opc2 = 4,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->pmsav7_dregion << 8

        };

        ARMCPRegInfo crn0_wi_reginfo = {

            .name = "CRN0_WI", .cp = 15, .crn = 0, .crm = CP_ANY,

            .opc1 = CP_ANY, .opc2 = CP_ANY, .access = PL1_W,

            .type = ARM_CP_NOP | ARM_CP_OVERRIDE

        };

        if (arm_feature(env, ARM_FEATURE_OMAPCP) ||

            arm_feature(env, ARM_FEATURE_STRONGARM)) {

            ARMCPRegInfo *r;

            /* Register the blanket "writes ignored" value first to cover the

             * whole space. Then update the specific ID registers to allow write

             * access, so that they ignore writes rather than causing them to

             * UNDEF.

             */

            define_one_arm_cp_reg(cpu, &crn0_wi_reginfo);

            for (r = id_pre_v8_midr_cp_reginfo;

                 r->type != ARM_CP_SENTINEL; r++) {

                r->access = PL1_RW;

            }

            for (r = id_cp_reginfo; r->type != ARM_CP_SENTINEL; r++) {

                r->access = PL1_RW;

            }

            id_tlbtr_reginfo.access = PL1_RW;

            id_tlbtr_reginfo.access = PL1_RW;

        }

        if (arm_feature(env, ARM_FEATURE_V8)) {

            define_arm_cp_regs(cpu, id_v8_midr_cp_reginfo);

        } else {

            define_arm_cp_regs(cpu, id_pre_v8_midr_cp_reginfo);

        }

        define_arm_cp_regs(cpu, id_cp_reginfo);

        if (!arm_feature(env, ARM_FEATURE_PMSA)) {

            define_one_arm_cp_reg(cpu, &id_tlbtr_reginfo);

        } else if (arm_feature(env, ARM_FEATURE_V7)) {

            define_one_arm_cp_reg(cpu, &id_mpuir_reginfo);

        }

    }



    if (arm_feature(env, ARM_FEATURE_MPIDR)) {

        define_arm_cp_regs(cpu, mpidr_cp_reginfo);

    }



    if (arm_feature(env, ARM_FEATURE_AUXCR)) {

        ARMCPRegInfo auxcr_reginfo[] = {

            { .name = "ACTLR_EL1", .state = ARM_CP_STATE_BOTH,

              .opc0 = 3, .opc1 = 0, .crn = 1, .crm = 0, .opc2 = 1,

              .access = PL1_RW, .type = ARM_CP_CONST,

              .resetvalue = cpu->reset_auxcr },

            { .name = "ACTLR_EL2", .state = ARM_CP_STATE_BOTH,

              .opc0 = 3, .opc1 = 4, .crn = 1, .crm = 0, .opc2 = 1,

              .access = PL2_RW, .type = ARM_CP_CONST,

              .resetvalue = 0 },

            { .name = "ACTLR_EL3", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 6, .crn = 1, .crm = 0, .opc2 = 1,

              .access = PL3_RW, .type = ARM_CP_CONST,

              .resetvalue = 0 },

            REGINFO_SENTINEL

        };

        define_arm_cp_regs(cpu, auxcr_reginfo);

    }



    if (arm_feature(env, ARM_FEATURE_CBAR)) {

        if (arm_feature(env, ARM_FEATURE_AARCH64)) {

            /* 32 bit view is [31:18] 0...0 [43:32]. */

            uint32_t cbar32 = (extract64(cpu->reset_cbar, 18, 14) << 18)

                | extract64(cpu->reset_cbar, 32, 12);

            ARMCPRegInfo cbar_reginfo[] = {

                { .name = "CBAR",

                  .type = ARM_CP_CONST,

                  .cp = 15, .crn = 15, .crm = 0, .opc1 = 4, .opc2 = 0,

                  .access = PL1_R, .resetvalue = cpu->reset_cbar },

                { .name = "CBAR_EL1", .state = ARM_CP_STATE_AA64,

                  .type = ARM_CP_CONST,

                  .opc0 = 3, .opc1 = 1, .crn = 15, .crm = 3, .opc2 = 0,

                  .access = PL1_R, .resetvalue = cbar32 },

                REGINFO_SENTINEL

            };

            /* We don't implement a r/w 64 bit CBAR currently */

            assert(arm_feature(env, ARM_FEATURE_CBAR_RO));

            define_arm_cp_regs(cpu, cbar_reginfo);

        } else {

            ARMCPRegInfo cbar = {

                .name = "CBAR",

                .cp = 15, .crn = 15, .crm = 0, .opc1 = 4, .opc2 = 0,

                .access = PL1_R|PL3_W, .resetvalue = cpu->reset_cbar,

                .fieldoffset = offsetof(CPUARMState,

                                        cp15.c15_config_base_address)

            };

            if (arm_feature(env, ARM_FEATURE_CBAR_RO)) {

                cbar.access = PL1_R;

                cbar.fieldoffset = 0;

                cbar.type = ARM_CP_CONST;

            }

            define_one_arm_cp_reg(cpu, &cbar);

        }

    }



    if (arm_feature(env, ARM_FEATURE_VBAR)) {

        ARMCPRegInfo vbar_cp_reginfo[] = {

            { .name = "VBAR", .state = ARM_CP_STATE_BOTH,

              .opc0 = 3, .crn = 12, .crm = 0, .opc1 = 0, .opc2 = 0,

              .access = PL1_RW, .writefn = vbar_write,

              .bank_fieldoffsets = { offsetof(CPUARMState, cp15.vbar_s),

                                     offsetof(CPUARMState, cp15.vbar_ns) },

              .resetvalue = 0 },

            REGINFO_SENTINEL

        };

        define_arm_cp_regs(cpu, vbar_cp_reginfo);

    }



    /* Generic registers whose values depend on the implementation */

    {

        ARMCPRegInfo sctlr = {

            .name = "SCTLR", .state = ARM_CP_STATE_BOTH,

            .opc0 = 3, .opc1 = 0, .crn = 1, .crm = 0, .opc2 = 0,

            .access = PL1_RW,

            .bank_fieldoffsets = { offsetof(CPUARMState, cp15.sctlr_s),

                                   offsetof(CPUARMState, cp15.sctlr_ns) },

            .writefn = sctlr_write, .resetvalue = cpu->reset_sctlr,

            .raw_writefn = raw_write,

        };

        if (arm_feature(env, ARM_FEATURE_XSCALE)) {

            /* Normally we would always end the TB on an SCTLR write, but Linux

             * arch/arm/mach-pxa/sleep.S expects two instructions following

             * an MMU enable to execute from cache.  Imitate this behaviour.

             */

            sctlr.type |= ARM_CP_SUPPRESS_TB_END;

        }

        define_one_arm_cp_reg(cpu, &sctlr);

    }

}
