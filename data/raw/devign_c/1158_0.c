static int ppc_hash32_pte_update_flags(struct mmu_ctx_hash32 *ctx, target_ulong *pte1p,

                                       int ret, int rwx)

{

    int store = 0;



    /* Update page flags */

    if (!(*pte1p & HPTE32_R_R)) {

        /* Update accessed flag */

        *pte1p |= HPTE32_R_R;

        store = 1;

    }

    if (!(*pte1p & HPTE32_R_C)) {

        if (rwx == 1 && ret == 0) {

            /* Update changed flag */

            *pte1p |= HPTE32_R_C;

            store = 1;

        } else {

            /* Force page fault for first write access */

            ctx->prot &= ~PAGE_WRITE;

        }

    }



    return store;

}
