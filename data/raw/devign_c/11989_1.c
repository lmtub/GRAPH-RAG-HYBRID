static void host_x86_cpu_class_init(ObjectClass *oc, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(oc);

    X86CPUClass *xcc = X86_CPU_CLASS(oc);

    uint32_t eax = 0, ebx = 0, ecx = 0, edx = 0;



    xcc->kvm_required = true;



    host_cpuid(0x0, 0, &eax, &ebx, &ecx, &edx);

    x86_cpu_vendor_words2str(host_cpudef.vendor, ebx, edx, ecx);



    host_cpuid(0x1, 0, &eax, &ebx, &ecx, &edx);

    host_cpudef.family = ((eax >> 8) & 0x0F) + ((eax >> 20) & 0xFF);

    host_cpudef.model = ((eax >> 4) & 0x0F) | ((eax & 0xF0000) >> 12);

    host_cpudef.stepping = eax & 0x0F;



    cpu_x86_fill_model_id(host_cpudef.model_id);



    xcc->cpu_def = &host_cpudef;

    host_cpudef.cache_info_passthrough = true;



    /* level, xlevel, xlevel2, and the feature words are initialized on

     * instance_init, because they require KVM to be initialized.

     */



    dc->props = host_x86_cpu_properties;



}