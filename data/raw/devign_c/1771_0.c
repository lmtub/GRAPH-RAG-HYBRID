uint32_t kvm_arch_get_supported_cpuid(KVMState *s, uint32_t function,

                                      uint32_t index, int reg)

{

    struct kvm_cpuid2 *cpuid;

    int max;

    uint32_t ret = 0;

    uint32_t cpuid_1_edx;

    bool found = false;



    max = 1;

    while ((cpuid = try_get_cpuid(s, max)) == NULL) {

        max *= 2;

    }



    struct kvm_cpuid_entry2 *entry = cpuid_find_entry(cpuid, function, index);

    if (entry) {

        found = true;

        ret = cpuid_entry_get_reg(entry, reg);

    }



    /* Fixups for the data returned by KVM, below */



    if (reg == R_EDX) {

        switch (function) {

        case 1:

            /* KVM before 2.6.30 misreports the following features */

            ret |= CPUID_MTRR | CPUID_PAT | CPUID_MCE | CPUID_MCA;

            break;

        case 0x80000001:

            /* On Intel, kvm returns cpuid according to the Intel spec,

             * so add missing bits according to the AMD spec:

             */

            cpuid_1_edx = kvm_arch_get_supported_cpuid(s, 1, 0, R_EDX);

            ret |= cpuid_1_edx & CPUID_EXT2_AMD_ALIASES;

            break;

        }

    }



    g_free(cpuid);



    /* fallback for older kernels */

    if ((function == KVM_CPUID_FEATURES) && !found) {

        ret = get_para_features(s);

    }



    return ret;

}
