int kvm_arch_on_sigbus(int code, void *addr)

{

#ifdef KVM_CAP_MCE

    if ((first_cpu->mcg_cap & MCG_SER_P) && addr && code == BUS_MCEERR_AO) {

        void *vaddr;

        ram_addr_t ram_addr;

        target_phys_addr_t paddr;



        /* Hope we are lucky for AO MCE */

        vaddr = addr;

        if (qemu_ram_addr_from_host(vaddr, &ram_addr) ||

            !kvm_physical_memory_addr_from_ram(first_cpu->kvm_state, ram_addr,

                                               &paddr)) {

            fprintf(stderr, "Hardware memory error for memory used by "

                    "QEMU itself instead of guest system!: %p\n", addr);

            return 0;

        }

        kvm_mce_inj_srao_memscrub2(first_cpu, paddr);

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
