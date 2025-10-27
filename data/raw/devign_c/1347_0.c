static bool riccb_needed(void *opaque)

{

#ifdef CONFIG_KVM

    if (kvm_enabled()) {

        return kvm_s390_get_ri();

    }

#endif

    return 0;

}
