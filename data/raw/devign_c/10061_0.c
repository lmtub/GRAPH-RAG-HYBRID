static void define_debug_regs(ARMCPU *cpu)

{

    /* Define v7 and v8 architectural debug registers.

     * These are just dummy implementations for now.

     */

    int i;



    define_arm_cp_regs(cpu, debug_cp_reginfo);



    if (arm_feature(&cpu->env, ARM_FEATURE_LPAE)) {

        define_arm_cp_regs(cpu, debug_lpae_cp_reginfo);

    }



    for (i = 0; i < 16; i++) {

        ARMCPRegInfo dbgregs[] = {

            { .name = "DBGBVR", .state = ARM_CP_STATE_BOTH,

              .cp = 14, .opc0 = 2, .opc1 = 0, .crn = 0, .crm = i, .opc2 = 4,

              .access = PL1_RW,

              .fieldoffset = offsetof(CPUARMState, cp15.dbgbvr[i]) },

            { .name = "DBGBCR", .state = ARM_CP_STATE_BOTH,

              .cp = 14, .opc0 = 2, .opc1 = 0, .crn = 0, .crm = i, .opc2 = 5,

              .access = PL1_RW,

              .fieldoffset = offsetof(CPUARMState, cp15.dbgbcr[i]) },

            { .name = "DBGWVR", .state = ARM_CP_STATE_BOTH,

              .cp = 14, .opc0 = 2, .opc1 = 0, .crn = 0, .crm = i, .opc2 = 6,

              .access = PL1_RW,

              .fieldoffset = offsetof(CPUARMState, cp15.dbgwvr[i]) },

            { .name = "DBGWCR", .state = ARM_CP_STATE_BOTH,

              .cp = 14, .opc0 = 2, .opc1 = 0, .crn = 0, .crm = i, .opc2 = 7,

              .access = PL1_RW,

              .fieldoffset = offsetof(CPUARMState, cp15.dbgwcr[i]) },

               REGINFO_SENTINEL

        };

        define_arm_cp_regs(cpu, dbgregs);

    }

}
