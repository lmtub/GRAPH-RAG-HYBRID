void kvmppc_hash64_write_pte(CPUPPCState *env, target_ulong pte_index,

                             target_ulong pte0, target_ulong pte1)

{

    int htab_fd;

    struct kvm_get_htab_fd ghf;

    struct kvm_get_htab_buf hpte_buf;



    ghf.flags = 0;

    ghf.start_index = 0;     /* Ignored */

    htab_fd = kvm_vm_ioctl(kvm_state, KVM_PPC_GET_HTAB_FD, &ghf);

    if (htab_fd < 0) {

        goto error_out;

    }



    hpte_buf.header.n_valid = 1;

    hpte_buf.header.n_invalid = 0;

    hpte_buf.header.index = pte_index;

    hpte_buf.hpte[0] = pte0;

    hpte_buf.hpte[1] = pte1;

    /*

     * Write the hpte entry.

     * CAUTION: write() has the warn_unused_result attribute. Hence we

     * need to check the return value, even though we do nothing.

     */

    if (write(htab_fd, &hpte_buf, sizeof(hpte_buf)) < 0) {

        goto out_close;

    }



out_close:

    close(htab_fd);

    return;



error_out:

    return;

}
