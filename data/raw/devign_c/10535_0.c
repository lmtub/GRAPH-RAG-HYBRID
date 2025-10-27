bool write_list_to_cpustate(ARMCPU *cpu)

{

    int i;

    bool ok = true;



    for (i = 0; i < cpu->cpreg_array_len; i++) {

        uint32_t regidx = kvm_to_cpreg_id(cpu->cpreg_indexes[i]);

        uint64_t v = cpu->cpreg_values[i];

        const ARMCPRegInfo *ri;



        ri = get_arm_cp_reginfo(cpu->cp_regs, regidx);

        if (!ri) {

            ok = false;

            continue;

        }

        if (ri->type & ARM_CP_NO_MIGRATE) {

            continue;

        }

        /* Write value and confirm it reads back as written

         * (to catch read-only registers and partially read-only

         * registers where the incoming migration value doesn't match)

         */

        write_raw_cp_reg(&cpu->env, ri, v);

        if (read_raw_cp_reg(&cpu->env, ri) != v) {

            ok = false;

        }

    }

    return ok;

}
