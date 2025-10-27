int mmubooke_get_physical_address (CPUState *env, mmu_ctx_t *ctx,

                                   target_ulong address, int rw,

                                   int access_type)

{

    ppcemb_tlb_t *tlb;

    target_phys_addr_t raddr;

    int i, prot, ret;



    ret = -1;

    raddr = -1;

    for (i = 0; i < env->nb_tlb; i++) {

        tlb = &env->tlb[i].tlbe;

        if (ppcemb_tlb_check(env, tlb, &raddr, address,

                             env->spr[SPR_BOOKE_PID], 1, i) < 0)

            continue;

        if (msr_pr != 0)

            prot = tlb->prot & 0xF;

        else

            prot = (tlb->prot >> 4) & 0xF;

        /* Check the address space */

        if (access_type == ACCESS_CODE) {

            if (msr_ir != (tlb->attr & 1))

                continue;

            ctx->prot = prot;

            if (prot & PAGE_EXEC) {

                ret = 0;

                break;

            }

            ret = -3;

        } else {

            if (msr_dr != (tlb->attr & 1))

                continue;

            ctx->prot = prot;

            if ((!rw && prot & PAGE_READ) || (rw && (prot & PAGE_WRITE))) {

                ret = 0;

                break;

            }

            ret = -2;

        }

    }

    if (ret >= 0)

        ctx->raddr = raddr;



    return ret;

}
