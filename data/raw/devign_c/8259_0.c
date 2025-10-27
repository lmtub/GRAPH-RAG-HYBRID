static void kvm_cpu_fill_host(x86_def_t *x86_cpu_def)

{

    uint32_t eax = 0, ebx = 0, ecx = 0, edx = 0;



    assert(kvm_enabled());



    x86_cpu_def->name = "host";

    host_cpuid(0x0, 0, &eax, &ebx, &ecx, &edx);

    x86_cpu_def->level = eax;

    x86_cpu_def->vendor1 = ebx;

    x86_cpu_def->vendor2 = edx;

    x86_cpu_def->vendor3 = ecx;



    host_cpuid(0x1, 0, &eax, &ebx, &ecx, &edx);

    x86_cpu_def->family = ((eax >> 8) & 0x0F) + ((eax >> 20) & 0xFF);

    x86_cpu_def->model = ((eax >> 4) & 0x0F) | ((eax & 0xF0000) >> 12);

    x86_cpu_def->stepping = eax & 0x0F;

    x86_cpu_def->ext_features = ecx;

    x86_cpu_def->features = edx;



    if (x86_cpu_def->level >= 7) {

        x86_cpu_def->cpuid_7_0_ebx_features = kvm_arch_get_supported_cpuid(kvm_state, 0x7, 0, R_EBX);

    } else {

        x86_cpu_def->cpuid_7_0_ebx_features = 0;

    }



    host_cpuid(0x80000000, 0, &eax, &ebx, &ecx, &edx);

    x86_cpu_def->xlevel = eax;



    host_cpuid(0x80000001, 0, &eax, &ebx, &ecx, &edx);

    x86_cpu_def->ext2_features = edx;

    x86_cpu_def->ext3_features = ecx;

    cpu_x86_fill_model_id(x86_cpu_def->model_id);

    x86_cpu_def->vendor_override = 0;



    /* Call Centaur's CPUID instruction. */

    if (x86_cpu_def->vendor1 == CPUID_VENDOR_VIA_1 &&

        x86_cpu_def->vendor2 == CPUID_VENDOR_VIA_2 &&

        x86_cpu_def->vendor3 == CPUID_VENDOR_VIA_3) {

        host_cpuid(0xC0000000, 0, &eax, &ebx, &ecx, &edx);

        if (eax >= 0xC0000001) {

            /* Support VIA max extended level */

            x86_cpu_def->xlevel2 = eax;

            host_cpuid(0xC0000001, 0, &eax, &ebx, &ecx, &edx);

            x86_cpu_def->ext4_features = edx;

        }

    }



    /*

     * Every SVM feature requires emulation support in KVM - so we can't just

     * read the host features here. KVM might even support SVM features not

     * available on the host hardware. Just set all bits and mask out the

     * unsupported ones later.

     */

    x86_cpu_def->svm_features = -1;

}
