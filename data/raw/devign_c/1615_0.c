int kvm_s390_set_mem_limit(KVMState *s, uint64_t new_limit, uint64_t *hw_limit)

{

    int rc;



    struct kvm_device_attr attr = {

        .group = KVM_S390_VM_MEM_CTRL,

        .attr = KVM_S390_VM_MEM_LIMIT_SIZE,

        .addr = (uint64_t) &new_limit,

    };



    if (!kvm_s390_supports_mem_limit(s)) {

        return 0;

    }



    rc = kvm_s390_query_mem_limit(s, hw_limit);

    if (rc) {

        return rc;

    } else if (*hw_limit < new_limit) {

        return -E2BIG;

    }



    return kvm_vm_ioctl(s, KVM_SET_DEVICE_ATTR, &attr);

}
