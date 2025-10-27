bool write_cpustate_to_list(ARMCPU *cpu)

{

    /* Write the coprocessor state from cpu->env to the (index,value) list. */

    int i;

    bool ok = true;



    for (i = 0; i < cpu->cpreg_array_len; i++) {

        uint32_t regidx = kvm_to_cpreg_id(cpu->cpreg_indexes[i]);

        const ARMCPRegInfo *ri;



        ri = get_arm_cp_reginfo(cpu->cp_regs, regidx);

        if (!ri) {

            ok = false;

            continue;

        }

        if (ri->type & ARM_CP_NO_MIGRATE) {

            continue;

        }

        cpu->cpreg_values[i] = read_raw_cp_reg(&cpu->env, ri);

    }

    return ok;

}
