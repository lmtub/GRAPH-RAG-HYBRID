static int kvm_irqchip_create(KVMState *s)

{

    QemuOptsList *list = qemu_find_opts("machine");

    int ret;



    if (QTAILQ_EMPTY(&list->head) ||

        !qemu_opt_get_bool(QTAILQ_FIRST(&list->head),

                           "kernel_irqchip", true) ||

        !kvm_check_extension(s, KVM_CAP_IRQCHIP)) {

        return 0;

    }



    ret = kvm_vm_ioctl(s, KVM_CREATE_IRQCHIP);

    if (ret < 0) {

        fprintf(stderr, "Create kernel irqchip failed\n");

        return ret;

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
