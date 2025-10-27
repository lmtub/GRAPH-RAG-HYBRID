static void cortex_a9_initfn(Object *obj)

{

    ARMCPU *cpu = ARM_CPU(obj);



    cpu->dtb_compatible = "arm,cortex-a9";

    set_feature(&cpu->env, ARM_FEATURE_V7);

    set_feature(&cpu->env, ARM_FEATURE_VFP3);

    set_feature(&cpu->env, ARM_FEATURE_VFP_FP16);

    set_feature(&cpu->env, ARM_FEATURE_NEON);

    set_feature(&cpu->env, ARM_FEATURE_THUMB2EE);

    /* Note that A9 supports the MP extensions even for

     * A9UP and single-core A9MP (which are both different

     * and valid configurations; we don't model A9UP).

     */

    set_feature(&cpu->env, ARM_FEATURE_V7MP);

    set_feature(&cpu->env, ARM_FEATURE_CBAR);

    cpu->midr = 0x410fc090;

    cpu->reset_fpsid = 0x41033090;

    cpu->mvfr0 = 0x11110222;

    cpu->mvfr1 = 0x01111111;

    cpu->ctr = 0x80038003;

    cpu->reset_sctlr = 0x00c50078;

    cpu->id_pfr0 = 0x1031;

    cpu->id_pfr1 = 0x11;

    cpu->id_dfr0 = 0x000;

    cpu->id_afr0 = 0;

    cpu->id_mmfr0 = 0x00100103;

    cpu->id_mmfr1 = 0x20000000;

    cpu->id_mmfr2 = 0x01230000;

    cpu->id_mmfr3 = 0x00002111;

    cpu->id_isar0 = 0x00101111;

    cpu->id_isar1 = 0x13112111;

    cpu->id_isar2 = 0x21232041;

    cpu->id_isar3 = 0x11112131;

    cpu->id_isar4 = 0x00111142;

    cpu->dbgdidr = 0x35141000;

    cpu->clidr = (1 << 27) | (1 << 24) | 3;

    cpu->ccsidr[0] = 0xe00fe015; /* 16k L1 dcache. */

    cpu->ccsidr[1] = 0x200fe015; /* 16k L1 icache. */

    define_arm_cp_regs(cpu, cortexa9_cp_reginfo);

}
