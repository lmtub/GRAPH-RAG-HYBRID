static int find_pte64(CPUPPCState *env, struct mmu_ctx_hash64 *ctx, int h,

                      int rw, int type, int target_page_bits)

{

    hwaddr pteg_off;

    target_ulong pte0, pte1;

    int i, good = -1;

    int ret, r;



    ret = -1; /* No entry found */

    pteg_off = (ctx->hash[h] * HASH_PTEG_SIZE_64) & env->htab_mask;

    for (i = 0; i < HPTES_PER_GROUP; i++) {

        pte0 = ppc_hash64_load_hpte0(env, pteg_off + i*HASH_PTE_SIZE_64);

        pte1 = ppc_hash64_load_hpte1(env, pteg_off + i*HASH_PTE_SIZE_64);



        r = pte64_check(ctx, pte0, pte1, h, rw, type);

        LOG_MMU("Load pte from %016" HWADDR_PRIx " => " TARGET_FMT_lx " "

                TARGET_FMT_lx " %d %d %d " TARGET_FMT_lx "\n",

                pteg_off + (i * 16), pte0, pte1, (int)(pte0 & 1), h,

                (int)((pte0 >> 1) & 1), ctx->ptem);

        switch (r) {

        case -3:

            /* PTE inconsistency */

            return -1;

        case -2:

            /* Access violation */

            ret = -2;

            good = i;

            break;

        case -1:

        default:

            /* No PTE match */

            break;

        case 0:

            /* access granted */

            /* XXX: we should go on looping to check all PTEs consistency

             *      but if we can speed-up the whole thing as the

             *      result would be undefined if PTEs are not consistent.

             */

            ret = 0;

            good = i;

            goto done;

        }

    }

    if (good != -1) {

    done:

        LOG_MMU("found PTE at addr %08" HWADDR_PRIx " prot=%01x ret=%d\n",

                ctx->raddr, ctx->prot, ret);

        /* Update page flags */

        pte1 = ctx->raddr;

        if (ppc_hash64_pte_update_flags(ctx, &pte1, ret, rw) == 1) {

            ppc_hash64_store_hpte1(env, pteg_off + good * HASH_PTE_SIZE_64, pte1);

        }

    }



    /* We have a TLB that saves 4K pages, so let's

     * split a huge page to 4k chunks */

    if (target_page_bits != TARGET_PAGE_BITS) {

        ctx->raddr |= (ctx->eaddr & ((1 << target_page_bits) - 1))

                      & TARGET_PAGE_MASK;

    }

    return ret;

}
