qemu_inject_x86_mce(CPUState *cenv, int bank, uint64_t status,

                    uint64_t mcg_status, uint64_t addr, uint64_t misc)

{

    uint64_t mcg_cap = cenv->mcg_cap;

    uint64_t *banks = cenv->mce_banks;



    /*

     * if MSR_MCG_CTL is not all 1s, the uncorrected error

     * reporting is disabled

     */

    if ((status & MCI_STATUS_UC) && (mcg_cap & MCG_CTL_P) &&

        cenv->mcg_ctl != ~(uint64_t)0) {

        return;

    }

    banks += 4 * bank;

    /*

     * if MSR_MCi_CTL is not all 1s, the uncorrected error

     * reporting is disabled for the bank

     */

    if ((status & MCI_STATUS_UC) && banks[0] != ~(uint64_t)0) {

        return;

    }

    if (status & MCI_STATUS_UC) {

        if ((cenv->mcg_status & MCG_STATUS_MCIP) ||

            !(cenv->cr[4] & CR4_MCE_MASK)) {

            fprintf(stderr, "injects mce exception while previous "

                    "one is in progress!\n");

            qemu_log_mask(CPU_LOG_RESET, "Triple fault\n");

            qemu_system_reset_request();

            return;

        }

        if (banks[1] & MCI_STATUS_VAL) {

            status |= MCI_STATUS_OVER;

        }

        banks[2] = addr;

        banks[3] = misc;

        cenv->mcg_status = mcg_status;

        banks[1] = status;

        cpu_interrupt(cenv, CPU_INTERRUPT_MCE);

    } else if (!(banks[1] & MCI_STATUS_VAL)

               || !(banks[1] & MCI_STATUS_UC)) {

        if (banks[1] & MCI_STATUS_VAL) {

            status |= MCI_STATUS_OVER;

        }

        banks[2] = addr;

        banks[3] = misc;

        banks[1] = status;

    } else {

        banks[1] |= MCI_STATUS_OVER;

    }

}
