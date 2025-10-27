static void vfio_add_kvm_msi_virq(VFIOMSIVector *vector, MSIMessage *msg,

                                  bool msix)

{

    int virq;



    if ((msix && !VFIO_ALLOW_KVM_MSIX) ||

        (!msix && !VFIO_ALLOW_KVM_MSI) || !msg) {

        return;

    }



    if (event_notifier_init(&vector->kvm_interrupt, 0)) {

        return;

    }



    virq = kvm_irqchip_add_msi_route(kvm_state, *msg);

    if (virq < 0) {

        event_notifier_cleanup(&vector->kvm_interrupt);

        return;

    }



    if (kvm_irqchip_add_irqfd_notifier_gsi(kvm_state, &vector->kvm_interrupt,

                                       NULL, virq) < 0) {

        kvm_irqchip_release_virq(kvm_state, virq);

        event_notifier_cleanup(&vector->kvm_interrupt);

        return;

    }



    vector->virq = virq;

}
