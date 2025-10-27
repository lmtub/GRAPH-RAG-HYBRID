int kvm_arch_init_vcpu(CPUState *env)

{

    struct {

        struct kvm_cpuid2 cpuid;

        struct kvm_cpuid_entry2 entries[100];

    } __attribute__((packed)) cpuid_data;

    uint32_t limit, i, j, cpuid_i;

    uint32_t unused;

    struct kvm_cpuid_entry2 *c;

    uint32_t signature[3];



    env->cpuid_features &= kvm_arch_get_supported_cpuid(env, 1, 0, R_EDX);



    i = env->cpuid_ext_features & CPUID_EXT_HYPERVISOR;

    env->cpuid_ext_features &= kvm_arch_get_supported_cpuid(env, 1, 0, R_ECX);

    env->cpuid_ext_features |= i;



    env->cpuid_ext2_features &= kvm_arch_get_supported_cpuid(env, 0x80000001,

                                                             0, R_EDX);

    env->cpuid_ext3_features &= kvm_arch_get_supported_cpuid(env, 0x80000001,

                                                             0, R_ECX);

    env->cpuid_svm_features  &= kvm_arch_get_supported_cpuid(env, 0x8000000A,

                                                             0, R_EDX);





    cpuid_i = 0;



    /* Paravirtualization CPUIDs */

    memcpy(signature, "KVMKVMKVM\0\0\0", 12);

    c = &cpuid_data.entries[cpuid_i++];

    memset(c, 0, sizeof(*c));

    c->function = KVM_CPUID_SIGNATURE;

    c->eax = 0;

    c->ebx = signature[0];

    c->ecx = signature[1];

    c->edx = signature[2];



    c = &cpuid_data.entries[cpuid_i++];

    memset(c, 0, sizeof(*c));

    c->function = KVM_CPUID_FEATURES;

    c->eax = env->cpuid_kvm_features & kvm_arch_get_supported_cpuid(env,

                                                KVM_CPUID_FEATURES, 0, R_EAX);



    has_msr_async_pf_en = c->eax & (1 << KVM_FEATURE_ASYNC_PF);



    cpu_x86_cpuid(env, 0, 0, &limit, &unused, &unused, &unused);



    for (i = 0; i <= limit; i++) {

        c = &cpuid_data.entries[cpuid_i++];



        switch (i) {

        case 2: {

            /* Keep reading function 2 till all the input is received */

            int times;



            c->function = i;

            c->flags = KVM_CPUID_FLAG_STATEFUL_FUNC |

                       KVM_CPUID_FLAG_STATE_READ_NEXT;

            cpu_x86_cpuid(env, i, 0, &c->eax, &c->ebx, &c->ecx, &c->edx);

            times = c->eax & 0xff;



            for (j = 1; j < times; ++j) {

                c = &cpuid_data.entries[cpuid_i++];

                c->function = i;

                c->flags = KVM_CPUID_FLAG_STATEFUL_FUNC;

                cpu_x86_cpuid(env, i, 0, &c->eax, &c->ebx, &c->ecx, &c->edx);

            }

            break;

        }

        case 4:

        case 0xb:

        case 0xd:

            for (j = 0; ; j++) {

                c->function = i;

                c->flags = KVM_CPUID_FLAG_SIGNIFCANT_INDEX;

                c->index = j;

                cpu_x86_cpuid(env, i, j, &c->eax, &c->ebx, &c->ecx, &c->edx);



                if (i == 4 && c->eax == 0) {

                    break;

                }

                if (i == 0xb && !(c->ecx & 0xff00)) {

                    break;

                }

                if (i == 0xd && c->eax == 0) {

                    break;

                }

                c = &cpuid_data.entries[cpuid_i++];

            }

            break;

        default:

            c->function = i;

            c->flags = 0;

            cpu_x86_cpuid(env, i, 0, &c->eax, &c->ebx, &c->ecx, &c->edx);

            break;

        }

    }

    cpu_x86_cpuid(env, 0x80000000, 0, &limit, &unused, &unused, &unused);



    for (i = 0x80000000; i <= limit; i++) {

        c = &cpuid_data.entries[cpuid_i++];



        c->function = i;

        c->flags = 0;

        cpu_x86_cpuid(env, i, 0, &c->eax, &c->ebx, &c->ecx, &c->edx);

    }



    /* Call Centaur's CPUID instructions they are supported. */

    if (env->cpuid_xlevel2 > 0) {

        env->cpuid_ext4_features &=

            kvm_arch_get_supported_cpuid(env, 0xC0000001, 0, R_EDX);

        cpu_x86_cpuid(env, 0xC0000000, 0, &limit, &unused, &unused, &unused);



        for (i = 0xC0000000; i <= limit; i++) {

            c = &cpuid_data.entries[cpuid_i++];



            c->function = i;

            c->flags = 0;

            cpu_x86_cpuid(env, i, 0, &c->eax, &c->ebx, &c->ecx, &c->edx);

        }

    }



    cpuid_data.cpuid.nent = cpuid_i;



    if (((env->cpuid_version >> 8)&0xF) >= 6

        && (env->cpuid_features&(CPUID_MCE|CPUID_MCA)) == (CPUID_MCE|CPUID_MCA)

        && kvm_check_extension(env->kvm_state, KVM_CAP_MCE) > 0) {

        uint64_t mcg_cap;

        int banks;

        int ret;



        ret = kvm_get_mce_cap_supported(env->kvm_state, &mcg_cap, &banks);

        if (ret < 0) {

            fprintf(stderr, "kvm_get_mce_cap_supported: %s", strerror(-ret));

            return ret;

        }



        if (banks > MCE_BANKS_DEF) {

            banks = MCE_BANKS_DEF;

        }

        mcg_cap &= MCE_CAP_DEF;

        mcg_cap |= banks;

        ret = kvm_vcpu_ioctl(env, KVM_X86_SETUP_MCE, &mcg_cap);

        if (ret < 0) {

            fprintf(stderr, "KVM_X86_SETUP_MCE: %s", strerror(-ret));

            return ret;

        }



        env->mcg_cap = mcg_cap;

    }



    qemu_add_vm_change_state_handler(cpu_update_state, env);



    return kvm_vcpu_ioctl(env, KVM_SET_CPUID2, &cpuid_data);

}
