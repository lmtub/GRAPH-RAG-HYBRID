static inline void ppc4xx_tlb_invalidate_virt(CPUPPCState *env,

                                              target_ulong eaddr, uint32_t pid)

{

#if !defined(FLUSH_ALL_TLBS)

    CPUState *cs = CPU(ppc_env_get_cpu(env));

    ppcemb_tlb_t *tlb;

    hwaddr raddr;

    target_ulong page, end;

    int i;



    for (i = 0; i < env->nb_tlb; i++) {

        tlb = &env->tlb.tlbe[i];

        if (ppcemb_tlb_check(env, tlb, &raddr, eaddr, pid, 0, i) == 0) {

            end = tlb->EPN + tlb->size;

            for (page = tlb->EPN; page < end; page += TARGET_PAGE_SIZE) {

                tlb_flush_page(cs, page);

            }

            tlb->prot &= ~PAGE_VALID;

            break;

        }

    }

#else

    ppc4xx_tlb_invalidate_all(env);

#endif

}
