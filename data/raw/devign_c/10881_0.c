int kvm_on_sigbus_vcpu(CPUState *env, int code, void *addr)

{

#if defined(KVM_CAP_MCE)

    struct kvm_x86_mce mce = {

            .bank = 9,

    };

    void *vaddr;

    ram_addr_t ram_addr;

    target_phys_addr_t paddr;

    int r;



    if ((env->mcg_cap & MCG_SER_P) && addr

        && (code == BUS_MCEERR_AR

            || code == BUS_MCEERR_AO)) {

        if (code == BUS_MCEERR_AR) {

            /* Fake an Intel architectural Data Load SRAR UCR */

            mce.status = MCI_STATUS_VAL | MCI_STATUS_UC | MCI_STATUS_EN

                | MCI_STATUS_MISCV | MCI_STATUS_ADDRV | MCI_STATUS_S

                | MCI_STATUS_AR | 0x134;

            mce.misc = (MCM_ADDR_PHYS << 6) | 0xc;

            mce.mcg_status = MCG_STATUS_MCIP | MCG_STATUS_EIPV;

        } else {

            /*

             * If there is an MCE excpetion being processed, ignore

             * this SRAO MCE

             */

            if (kvm_mce_in_progress(env)) {

                return 0;

            }

            /* Fake an Intel architectural Memory scrubbing UCR */

            mce.status = MCI_STATUS_VAL | MCI_STATUS_UC | MCI_STATUS_EN

                | MCI_STATUS_MISCV | MCI_STATUS_ADDRV | MCI_STATUS_S

                | 0xc0;

            mce.misc = (MCM_ADDR_PHYS << 6) | 0xc;

            mce.mcg_status = MCG_STATUS_MCIP | MCG_STATUS_RIPV;

        }

        vaddr = (void *)addr;

        if (qemu_ram_addr_from_host(vaddr, &ram_addr) ||

            !kvm_physical_memory_addr_from_ram(env->kvm_state, ram_addr, &paddr)) {

            fprintf(stderr, "Hardware memory error for memory used by "

                    "QEMU itself instead of guest system!\n");

            /* Hope we are lucky for AO MCE */

            if (code == BUS_MCEERR_AO) {

                return 0;

            } else {

                hardware_memory_error();

            }

        }

        mce.addr = paddr;

        r = kvm_set_mce(env, &mce);

        if (r < 0) {

            fprintf(stderr, "kvm_set_mce: %s\n", strerror(errno));

            abort();

        }

        kvm_mce_broadcast_rest(env);

    } else

#endif

    {

        if (code == BUS_MCEERR_AO) {

            return 0;

        } else if (code == BUS_MCEERR_AR) {

            hardware_memory_error();

        } else {

            return 1;

        }

    }

    return 0;

}
