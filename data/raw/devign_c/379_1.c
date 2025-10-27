static int kvm_recommended_vcpus(KVMState *s)

{

    int ret = kvm_check_extension(s, KVM_CAP_NR_VCPUS);

    return (ret) ? ret : 4;

}
