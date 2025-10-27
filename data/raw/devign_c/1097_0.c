static void count_cpreg(gpointer key, gpointer opaque)

{

    ARMCPU *cpu = opaque;

    uint64_t regidx;

    const ARMCPRegInfo *ri;



    regidx = *(uint32_t *)key;

    ri = get_arm_cp_reginfo(cpu->cp_regs, regidx);



    if (!(ri->type & ARM_CP_NO_MIGRATE)) {

        cpu->cpreg_array_len++;

    }

}
