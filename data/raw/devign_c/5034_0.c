static void arm1026_initfn(Object *obj)

{

    ARMCPU *cpu = ARM_CPU(obj);



    cpu->dtb_compatible = "arm,arm1026";

    set_feature(&cpu->env, ARM_FEATURE_V5);

    set_feature(&cpu->env, ARM_FEATURE_VFP);

    set_feature(&cpu->env, ARM_FEATURE_AUXCR);

    set_feature(&cpu->env, ARM_FEATURE_DUMMY_C15_REGS);

    set_feature(&cpu->env, ARM_FEATURE_CACHE_TEST_CLEAN);

    cpu->midr = 0x4106a262;

    cpu->reset_fpsid = 0x410110a0;

    cpu->ctr = 0x1dd20d2;

    cpu->reset_sctlr = 0x00090078;

    cpu->reset_auxcr = 1;

    {

        /* The 1026 had an IFAR at c6,c0,0,1 rather than the ARMv6 c6,c0,0,2 */

        ARMCPRegInfo ifar = {

            .name = "IFAR", .cp = 15, .crn = 6, .crm = 0, .opc1 = 0, .opc2 = 1,

            .access = PL1_RW,

            .fieldoffset = offsetofhigh32(CPUARMState, cp15.far_el[1]),

            .resetvalue = 0

        };

        define_one_arm_cp_reg(cpu, &ifar);

    }

}
