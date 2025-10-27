int kvmppc_reset_htab(int shift_hint)

{

    uint32_t shift = shift_hint;



    if (!kvm_enabled()) {

        /* Full emulation, tell caller to allocate htab itself */

        return 0;

    }

    if (kvm_check_extension(kvm_state, KVM_CAP_PPC_ALLOC_HTAB)) {

        int ret;

        ret = kvm_vm_ioctl(kvm_state, KVM_PPC_ALLOCATE_HTAB, &shift);

        if (ret == -ENOTTY) {

            /* At least some versions of PR KVM advertise the

             * capability, but don't implement the ioctl().  Oops.

             * Return 0 so that we allocate the htab in qemu, as is

             * correct for PR. */

            return 0;

        } else if (ret < 0) {

            return ret;

        }

        return shift;

    }



    /* We have a kernel that predates the htab reset calls.  For PR

     * KVM, we need to allocate the htab ourselves, for an HV KVM of

     * this era, it has allocated a 16MB fixed size hash table already. */

    if (kvmppc_is_pr(kvm_state)) {

        /* PR - tell caller to allocate htab */

        return 0;

    } else {

        /* HV - assume 16MB kernel allocated htab */

        return 24;

    }

}
