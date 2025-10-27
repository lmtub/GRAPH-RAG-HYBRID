uint64_t ppc_hash64_start_access(PowerPCCPU *cpu, target_ulong pte_index)

{

    uint64_t token = 0;

    hwaddr pte_offset;



    pte_offset = pte_index * HASH_PTE_SIZE_64;

    if (cpu->env.external_htab == MMU_HASH64_KVM_MANAGED_HPT) {

        /*

         * HTAB is controlled by KVM. Fetch the PTEG into a new buffer.

         */

        token = kvmppc_hash64_read_pteg(cpu, pte_index);

    } else if (cpu->env.external_htab) {

        /*

         * HTAB is controlled by QEMU. Just point to the internally

         * accessible PTEG.

         */

        token = (uint64_t)(uintptr_t) cpu->env.external_htab + pte_offset;

    } else if (cpu->env.htab_base) {

        token = cpu->env.htab_base + pte_offset;

    }

    return token;

}
