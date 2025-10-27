int kvm_arch_on_sigbus_vcpu(CPUState *env, int code, void *addr)

{

#ifdef KVM_CAP_MCE

    void *vaddr;

    ram_addr_t ram_addr;

    target_phys_addr_t paddr;



    if ((env->mcg_cap & MCG_SER_P) && addr

        && (code == BUS_MCEERR_AR

            || code == BUS_MCEERR_AO)) {

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



        if (code == BUS_MCEERR_AR) {

            /* Fake an Intel architectural Data Load SRAR UCR */

            kvm_mce_inj_srar_dataload(env, paddr);

        } else {

            /*

             * If there is an MCE excpetion being processed, ignore

             * this SRAO MCE

             */

            if (!kvm_mce_in_progress(env)) {

                /* Fake an Intel architectural Memory scrubbing UCR */

                kvm_mce_inj_srao_memscrub(env, paddr);

            }

        }

    } else

#endif /* KVM_CAP_MCE */

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
