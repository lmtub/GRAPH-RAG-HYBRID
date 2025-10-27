void kvm_s390_cmma_reset(void)

{

    int rc;

    struct kvm_device_attr attr = {

        .group = KVM_S390_VM_MEM_CTRL,

        .attr = KVM_S390_VM_MEM_CLR_CMMA,

    };



    if (mem_path || !kvm_s390_cmma_available()) {

        return;

    }



    rc = kvm_vm_ioctl(kvm_state, KVM_SET_DEVICE_ATTR, &attr);

    trace_kvm_clear_cmma(rc);

}
