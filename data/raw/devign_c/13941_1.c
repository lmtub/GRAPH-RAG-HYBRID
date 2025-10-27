int kvm_has_sync_mmu(void)

{

#ifdef KVM_CAP_SYNC_MMU

    KVMState *s = kvm_state;



    return kvm_check_extension(s, KVM_CAP_SYNC_MMU);

#else

    return 0;

#endif

}
