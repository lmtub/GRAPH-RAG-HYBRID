void ppc_hash64_stop_access(uint64_t token)

{

    if (kvmppc_kern_htab) {

        kvmppc_hash64_free_pteg(token);

    }

}
