void ppc_hash64_store_hpte(PowerPCCPU *cpu,

                           target_ulong pte_index,

                           target_ulong pte0, target_ulong pte1)

{

    CPUPPCState *env = &cpu->env;



    if (env->external_htab == MMU_HASH64_KVM_MANAGED_HPT) {

        kvmppc_hash64_write_pte(env, pte_index, pte0, pte1);

        return;

    }



    pte_index *= HASH_PTE_SIZE_64;

    if (env->external_htab) {

        stq_p(env->external_htab + pte_index, pte0);

        stq_p(env->external_htab + pte_index + HASH_PTE_SIZE_64 / 2, pte1);

    } else {

        stq_phys(CPU(cpu)->as, env->htab_base + pte_index, pte0);

        stq_phys(CPU(cpu)->as,

                 env->htab_base + pte_index + HASH_PTE_SIZE_64 / 2, pte1);

    }

}
