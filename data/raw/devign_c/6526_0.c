static void get_cpuid_vendor(CPUX86State *env, uint32_t *ebx,

                             uint32_t *ecx, uint32_t *edx)

{

    *ebx = env->cpuid_vendor1;

    *edx = env->cpuid_vendor2;

    *ecx = env->cpuid_vendor3;



    /* sysenter isn't supported on compatibility mode on AMD, syscall

     * isn't supported in compatibility mode on Intel.

     * Normally we advertise the actual cpu vendor, but you can override

     * this if you want to use KVM's sysenter/syscall emulation

     * in compatibility mode and when doing cross vendor migration

     */

    if (kvm_enabled() && env->cpuid_vendor_override) {

        host_cpuid(0, 0, NULL, ebx, ecx, edx);

    }

}
