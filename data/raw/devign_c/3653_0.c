static int kvm_has_msr_star(CPUState *env)

{

    kvm_supported_msrs(env);

    return has_msr_star;

}
