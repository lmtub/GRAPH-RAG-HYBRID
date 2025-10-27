static bool gscb_needed(void *opaque)

{

    return kvm_s390_get_gs();

}
