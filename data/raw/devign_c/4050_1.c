static bool check_irqchip_in_kernel(void)

{

    if (kvm_irqchip_in_kernel()) {

        return true;

    }

    error_report("pci-assign: error: requires KVM with in-kernel irqchip "

                 "enabled");

    return false;

}
