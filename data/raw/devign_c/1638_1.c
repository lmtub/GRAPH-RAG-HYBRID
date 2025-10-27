int kvm_uncoalesce_mmio_region(target_phys_addr_t start, ram_addr_t size)

{

    int ret = -ENOSYS;

    KVMState *s = kvm_state;



    if (s->coalesced_mmio) {

        struct kvm_coalesced_mmio_zone zone;



        zone.addr = start;

        zone.size = size;




        ret = kvm_vm_ioctl(s, KVM_UNREGISTER_COALESCED_MMIO, &zone);

    }



    return ret;

}