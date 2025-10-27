int kvm_irqchip_add_irqfd(KVMState *s, int fd, int virq)

{

    return kvm_irqchip_assign_irqfd(s, fd, virq, true);

}
