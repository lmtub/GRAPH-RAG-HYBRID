static void xics_kvm_cpu_setup(XICSState *xics, PowerPCCPU *cpu)

{

    CPUState *cs;

    ICPState *ss;

    KVMXICSState *xicskvm = XICS_SPAPR_KVM(xics);



    cs = CPU(cpu);

    ss = &xics->ss[cs->cpu_index];



    assert(cs->cpu_index < xics->nr_servers);

    if (xicskvm->kernel_xics_fd == -1) {

        abort();

    }



    /*

     * If we are reusing a parked vCPU fd corresponding to the CPU

     * which was hot-removed earlier we don't have to renable

     * KVM_CAP_IRQ_XICS capability again.

     */

    if (ss->cap_irq_xics_enabled) {

        return;

    }



    if (xicskvm->kernel_xics_fd != -1) {

        int ret;



        ret = kvm_vcpu_enable_cap(cs, KVM_CAP_IRQ_XICS, 0,

                                  xicskvm->kernel_xics_fd,

                                  kvm_arch_vcpu_id(cs));

        if (ret < 0) {

            error_report("Unable to connect CPU%ld to kernel XICS: %s",

                    kvm_arch_vcpu_id(cs), strerror(errno));

            exit(1);

        }

        ss->cap_irq_xics_enabled = true;

    }

}
