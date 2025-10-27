static int hyperv_handle_properties(CPUState *cs)

{

    X86CPU *cpu = X86_CPU(cs);

    CPUX86State *env = &cpu->env;



    if (cpu->hyperv_relaxed_timing) {

        env->features[FEAT_HYPERV_EAX] |= HV_X64_MSR_HYPERCALL_AVAILABLE;

    }

    if (cpu->hyperv_vapic) {

        env->features[FEAT_HYPERV_EAX] |= HV_X64_MSR_HYPERCALL_AVAILABLE;

        env->features[FEAT_HYPERV_EAX] |= HV_X64_MSR_APIC_ACCESS_AVAILABLE;

    }

    if (cpu->hyperv_time &&

            kvm_check_extension(cs->kvm_state, KVM_CAP_HYPERV_TIME) > 0) {

        env->features[FEAT_HYPERV_EAX] |= HV_X64_MSR_HYPERCALL_AVAILABLE;

        env->features[FEAT_HYPERV_EAX] |= HV_X64_MSR_TIME_REF_COUNT_AVAILABLE;

        env->features[FEAT_HYPERV_EAX] |= 0x200;

        has_msr_hv_tsc = true;

    }

    if (cpu->hyperv_crash && has_msr_hv_crash) {

        env->features[FEAT_HYPERV_EDX] |= HV_X64_GUEST_CRASH_MSR_AVAILABLE;

    }

    env->features[FEAT_HYPERV_EDX] |= HV_X64_CPU_DYNAMIC_PARTITIONING_AVAILABLE;

    if (cpu->hyperv_reset && has_msr_hv_reset) {

        env->features[FEAT_HYPERV_EAX] |= HV_X64_MSR_RESET_AVAILABLE;

    }

    if (cpu->hyperv_vpindex && has_msr_hv_vpindex) {

        env->features[FEAT_HYPERV_EAX] |= HV_X64_MSR_VP_INDEX_AVAILABLE;

    }

    if (cpu->hyperv_runtime && has_msr_hv_runtime) {

        env->features[FEAT_HYPERV_EAX] |= HV_X64_MSR_VP_RUNTIME_AVAILABLE;

    }

    if (cpu->hyperv_synic) {

        int sint;



        if (!has_msr_hv_synic ||

            kvm_vcpu_enable_cap(cs, KVM_CAP_HYPERV_SYNIC, 0)) {

            fprintf(stderr, "Hyper-V SynIC is not supported by kernel\n");

            return -ENOSYS;

        }



        env->features[FEAT_HYPERV_EAX] |= HV_X64_MSR_SYNIC_AVAILABLE;

        env->msr_hv_synic_version = HV_SYNIC_VERSION_1;

        for (sint = 0; sint < ARRAY_SIZE(env->msr_hv_synic_sint); sint++) {

            env->msr_hv_synic_sint[sint] = HV_SYNIC_SINT_MASKED;

        }

    }

    if (cpu->hyperv_stimer) {

        if (!has_msr_hv_stimer) {

            fprintf(stderr, "Hyper-V timers aren't supported by kernel\n");

            return -ENOSYS;

        }

        env->features[FEAT_HYPERV_EAX] |= HV_X64_MSR_SYNTIMER_AVAILABLE;

    }

    return 0;

}
