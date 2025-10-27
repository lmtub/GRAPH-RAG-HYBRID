static uint64_t read_raw_cp_reg(CPUARMState *env, const ARMCPRegInfo *ri)

{

    /* Raw read of a coprocessor register (as needed for migration, etc). */

    if (ri->type & ARM_CP_CONST) {

        return ri->resetvalue;

    } else if (ri->raw_readfn) {

        return ri->raw_readfn(env, ri);

    } else if (ri->readfn) {

        return ri->readfn(env, ri);

    } else {

        return raw_read(env, ri);

    }

}
