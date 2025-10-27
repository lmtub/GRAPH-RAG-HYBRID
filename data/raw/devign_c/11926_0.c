static int pte64_check(struct mmu_ctx_hash64 *ctx, target_ulong pte0,

                       target_ulong pte1, int h, int rw, int type)

{

    target_ulong mmask;

    int access, ret, pp;



    ret = -1;

    /* Check validity and table match */

    if ((pte0 & HPTE64_V_VALID) && (h == !!(pte0 & HPTE64_V_SECONDARY))) {

        /* Check vsid & api */

        mmask = PTE64_CHECK_MASK;

        pp = (pte1 & HPTE64_R_PP) | ((pte1 & HPTE64_R_PP0) >> 61);

        /* No execute if either noexec or guarded bits set */

        ctx->nx = (pte1 & HPTE64_R_N) || (pte1 & HPTE64_R_G);

        if (HPTE64_V_COMPARE(pte0, ctx->ptem)) {

            if (ctx->raddr != (hwaddr)-1ULL) {

                /* all matches should have equal RPN, WIMG & PP */

                if ((ctx->raddr & mmask) != (pte1 & mmask)) {

                    qemu_log("Bad RPN/WIMG/PP\n");

                    return -3;

                }

            }

            /* Compute access rights */

            access = ppc_hash64_pp_check(ctx->key, pp, ctx->nx);

            /* Keep the matching PTE informations */

            ctx->raddr = pte1;

            ctx->prot = access;

            ret = ppc_hash64_check_prot(ctx->prot, rw, type);

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
