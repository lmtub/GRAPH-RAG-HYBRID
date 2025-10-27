uint64_t kvmppc_hash64_read_pteg(PowerPCCPU *cpu, target_ulong pte_index)

{

    int htab_fd;

    struct kvm_get_htab_fd ghf;

    struct kvm_get_htab_buf  *hpte_buf;



    ghf.flags = 0;

    ghf.start_index = pte_index;

    htab_fd = kvm_vm_ioctl(kvm_state, KVM_PPC_GET_HTAB_FD, &ghf);

    if (htab_fd < 0) {

        goto error_out;

    }



    hpte_buf = g_malloc0(sizeof(*hpte_buf));

    /*

     * Read the hpte group

     */

    if (read(htab_fd, hpte_buf, sizeof(*hpte_buf)) < 0) {

        goto out_close;

    }



    close(htab_fd);

    return (uint64_t)(uintptr_t) hpte_buf->hpte;



out_close:

    g_free(hpte_buf);

    close(htab_fd);

error_out:

    return 0;

}
