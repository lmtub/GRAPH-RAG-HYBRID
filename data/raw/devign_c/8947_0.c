int kvm_arch_irqchip_create(MachineState *ms, KVMState *s)

{

    int ret;

    if (machine_kernel_irqchip_split(ms)) {

        ret = kvm_vm_enable_cap(s, KVM_CAP_SPLIT_IRQCHIP, 0, 24);

        if (ret) {

            error_report("Could not enable split irqchip mode: %s\n",

                         strerror(-ret));

            exit(1);

        } else {

            DPRINTF("Enabled KVM_CAP_SPLIT_IRQCHIP\n");

            kvm_split_irqchip = true;

            return 1;

        }

    } else {

        return 0;

    }

}
