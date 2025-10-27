void hmp_info_io_apic(Monitor *mon, const QDict *qdict)

{

    if (kvm_irqchip_in_kernel()) {

        kvm_ioapic_dump_state(mon, qdict);

    } else {

        ioapic_dump_state(mon, qdict);

    }

}
