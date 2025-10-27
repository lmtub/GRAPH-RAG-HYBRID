void ppc_hash64_stop_access(PowerPCCPU *cpu, uint64_t token)

{

    if (cpu->env.external_htab == MMU_HASH64_KVM_MANAGED_HPT) {

        kvmppc_hash64_free_pteg(token);

    }

}
