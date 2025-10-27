static int kvm_irqchip_create(MachineState *machine, KVMState *s)

{

    int ret;



    if (!machine_kernel_irqchip_allowed(machine) ||

        (!kvm_check_extension(s, KVM_CAP_IRQCHIP) &&

         (kvm_vm_enable_cap(s, KVM_CAP_S390_IRQCHIP, 0) < 0))) {

        return 0;

    }



    /* First probe and see if there's a arch-specific hook to create the

     * in-kernel irqchip for us */

    ret = kvm_arch_irqchip_create(s);

    if (ret < 0) {

        return ret;

    } else if (ret == 0) {

        ret = kvm_vm_ioctl(s, KVM_CREATE_IRQCHIP);

        if (ret < 0) {

            fprintf(stderr, "Create kernel irqchip failed\n");

            return ret;

        }

    }



    kvm_kernel_irqchip = true;

    /* If we have an in-kernel IRQ chip then we must have asynchronous

     * interrupt delivery (though the reverse is not necessarily true)

     */

    kvm_async_interrupts_allowed = true;

    kvm_halt_in_kernel_allowed = true;



    kvm_init_irq_routing(s);



    return 0;

}
