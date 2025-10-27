static int find_pte64(CPUPPCState *env, struct mmu_ctx_hash64 *ctx,

                      target_ulong eaddr, int h, int rwx, int target_page_bits)

{

    hwaddr pteg_off;

    target_ulong pte0, pte1;

    int i, good = -1;

    int ret;



    ret = -1; /* No entry found */

    pteg_off = (ctx->hash[h] * HASH_PTEG_SIZE_64) & env->htab_mask;

    for (i = 0; i < HPTES_PER_GROUP; i++) {

        pte0 = ppc_hash64_load_hpte0(env, pteg_off + i*HASH_PTE_SIZE_64);

        pte1 = ppc_hash64_load_hpte1(env, pteg_off + i*HASH_PTE_SIZE_64);



        LOG_MMU("Load pte from %016" HWADDR_PRIx " => " TARGET_FMT_lx " "

                TARGET_FMT_lx " %d %d %d " TARGET_FMT_lx "\n",

                pteg_off + (i * 16), pte0, pte1, !!(pte0 & HPTE64_V_VALID),

                h, !!(pte0 & HPTE64_V_SECONDARY), ctx->ptem);



        if (pte64_match(pte0, pte1, h, ctx->ptem)) {

            good = i;

            break;

        }

    }

    if (good != -1) {

        ret = pte64_check(ctx, pte0, pte1, rwx);

        LOG_MMU("found PTE at addr %08" HWADDR_PRIx " prot=%01x ret=%d\n",

                ctx->raddr, ctx->prot, ret);

        /* Update page flags */

        pte1 = ctx->raddr;

        if (ppc_hash64_pte_update_flags(ctx, &pte1, ret, rwx) == 1) {

            ppc_hash64_store_hpte1(env, pteg_off + good * HASH_PTE_SIZE_64, pte1);

        }

    }



    /* We have a TLB that saves 4K pages, so let's

     * split a huge page to 4k chunks */

    if (target_page_bits != TARGET_PAGE_BITS) {

        ctx->raddr |= (eaddr & ((1 << target_page_bits) - 1))

                      & TARGET_PAGE_MASK;

    }

    return ret;

}
