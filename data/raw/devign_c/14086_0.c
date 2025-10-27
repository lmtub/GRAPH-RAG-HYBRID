static void x86_cpuid_get_apic_id(Object *obj, Visitor *v, const char *name,

                                  void *opaque, Error **errp)

{

    X86CPU *cpu = X86_CPU(obj);

    int64_t value = cpu->apic_id;



    visit_type_int(v, name, &value, errp);

}
