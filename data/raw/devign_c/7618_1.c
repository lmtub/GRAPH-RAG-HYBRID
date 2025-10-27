void kvm_ioapic_dump_state(Monitor *mon, const QDict *qdict)

{

    IOAPICCommonState s;



    kvm_ioapic_get(&s);



    ioapic_print_redtbl(mon, &s);

}
