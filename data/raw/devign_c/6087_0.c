static int kvm_s390_check_clear_cmma(KVMState *s)

{

    struct kvm_device_attr attr = {

        .group = KVM_S390_VM_MEM_CTRL,

        .attr = KVM_S390_VM_MEM_CLR_CMMA,

    };



    return kvm_vm_ioctl(s, KVM_HAS_DEVICE_ATTR, &attr);

}
