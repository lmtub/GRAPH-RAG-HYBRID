static int pte_check_hash32(struct mmu_ctx_hash32 *ctx, target_ulong pte0,

                            target_ulong pte1, int h, int rw, int type)

{

    target_ulong mmask;

    int access, ret, pp;



    ret = -1;

    /* Check validity and table match */

    if ((pte0 & HPTE32_V_VALID) && (h == !!(pte0 & HPTE32_V_SECONDARY))) {

        /* Check vsid & api */

        mmask = PTE_CHECK_MASK;

        pp = pte1 & HPTE32_R_PP;

        if (HPTE32_V_COMPARE(pte0, ctx->ptem)) {

            if (ctx->raddr != (hwaddr)-1ULL) {

                /* all matches should have equal RPN, WIMG & PP */

                if ((ctx->raddr & mmask) != (pte1 & mmask)) {

                    qemu_log("Bad RPN/WIMG/PP\n");

                    return -3;

                }

            }

            /* Compute access rights */

            access = ppc_hash32_pp_check(ctx->key, pp, ctx->nx);

            /* Keep the matching PTE informations */

            ctx->raddr = pte1;

            ctx->prot = access;

            ret = ppc_hash32_check_prot(ctx->prot, rw, type);

            if (ret == 0) {

                /* Access granted */

                LOG_MMU("PTE access granted !\n");

            } else {

                /* Access right violation */

                LOG_MMU("PTE access rejected\n");

            }

        }

    }



    return ret;

}
