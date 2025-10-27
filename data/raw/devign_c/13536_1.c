static void arm926_initfn(Object *obj)

{

    ARMCPU *cpu = ARM_CPU(obj);



    cpu->dtb_compatible = "arm,arm926";

    set_feature(&cpu->env, ARM_FEATURE_V5);

    set_feature(&cpu->env, ARM_FEATURE_VFP);

    set_feature(&cpu->env, ARM_FEATURE_DUMMY_C15_REGS);

    set_feature(&cpu->env, ARM_FEATURE_CACHE_TEST_CLEAN);


    cpu->midr = 0x41069265;

    cpu->reset_fpsid = 0x41011090;

    cpu->ctr = 0x1dd20d2;

    cpu->reset_sctlr = 0x00090078;

}