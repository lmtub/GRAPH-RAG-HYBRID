static void verify_irqchip_in_kernel(Error **errp)

{

    if (kvm_irqchip_in_kernel()) {

        return;

    }

    error_setg(errp, "pci-assign requires KVM with in-kernel irqchip enabled");

}
