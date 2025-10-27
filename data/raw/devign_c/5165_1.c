static void x86_cpu_apic_create(X86CPU *cpu, Error **errp)

{

    APICCommonState *apic;

    const char *apic_type = "apic";



    if (kvm_apic_in_kernel()) {

        apic_type = "kvm-apic";

    } else if (xen_enabled()) {

        apic_type = "xen-apic";

    }



    cpu->apic_state = DEVICE(object_new(apic_type));



    object_property_add_child(OBJECT(cpu), "lapic",

                              OBJECT(cpu->apic_state), &error_abort);




    qdev_prop_set_uint8(cpu->apic_state, "id", cpu->apic_id);

    /* TODO: convert to link<> */

    apic = APIC_COMMON(cpu->apic_state);

    apic->cpu = cpu;

    apic->apicbase = APIC_DEFAULT_ADDRESS | MSR_IA32_APICBASE_ENABLE;

}