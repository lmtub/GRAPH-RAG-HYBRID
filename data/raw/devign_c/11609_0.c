static void x86_cpu_apic_create(X86CPU *cpu, Error **errp)

{

    DeviceState *dev = DEVICE(cpu);

    APICCommonState *apic;

    const char *apic_type = "apic";



    if (kvm_irqchip_in_kernel()) {

        apic_type = "kvm-apic";

    } else if (xen_enabled()) {

        apic_type = "xen-apic";

    }



    cpu->apic_state = qdev_try_create(qdev_get_parent_bus(dev), apic_type);

    if (cpu->apic_state == NULL) {

        error_setg(errp, "APIC device '%s' could not be created", apic_type);

        return;

    }



    object_property_add_child(OBJECT(cpu), "apic",

                              OBJECT(cpu->apic_state), NULL);

    qdev_prop_set_uint8(cpu->apic_state, "id", cpu->apic_id);

    /* TODO: convert to link<> */

    apic = APIC_COMMON(cpu->apic_state);

    apic->cpu = cpu;

    apic->apicbase = APIC_DEFAULT_ADDRESS | MSR_IA32_APICBASE_ENABLE;

}
