static bool hyperv_enabled(X86CPU *cpu)

{

    CPUState *cs = CPU(cpu);

    return kvm_check_extension(cs->kvm_state, KVM_CAP_HYPERV) > 0 &&

           (hyperv_hypercall_available(cpu) ||

            cpu->hyperv_time  ||

            cpu->hyperv_relaxed_timing);

}
