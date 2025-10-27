static int kvm_arch_sync_sregs(CPUState *cenv)

{

    struct kvm_sregs sregs;

    int ret;



    if (cenv->excp_model == POWERPC_EXCP_BOOKE) {

        return 0;

    } else {

        if (!cap_segstate) {

            return 0;

        }

    }



    ret = kvm_vcpu_ioctl(cenv, KVM_GET_SREGS, &sregs);

    if (ret) {

        return ret;

    }



    sregs.pvr = cenv->spr[SPR_PVR];

    return kvm_vcpu_ioctl(cenv, KVM_SET_SREGS, &sregs);

}
