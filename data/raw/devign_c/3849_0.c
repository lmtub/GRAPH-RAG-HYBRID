void ppc_hash64_store_hpte(PowerPCCPU *cpu, hwaddr ptex,

                           uint64_t pte0, uint64_t pte1)

{

    CPUPPCState *env = &cpu->env;

    hwaddr offset = ptex * HASH_PTE_SIZE_64;



    if (env->external_htab == MMU_HASH64_KVM_MANAGED_HPT) {

        kvmppc_write_hpte(ptex, pte0, pte1);

        return;

    }



    if (env->external_htab) {

        stq_p(env->external_htab + offset, pte0);

        stq_p(env->external_htab + offset + HASH_PTE_SIZE_64 / 2, pte1);

    } else {

        hwaddr base = ppc_hash64_hpt_base(cpu);

        stq_phys(CPU(cpu)->as, base + offset, pte0);

        stq_phys(CPU(cpu)->as, base + offset + HASH_PTE_SIZE_64 / 2, pte1);

    }

}
