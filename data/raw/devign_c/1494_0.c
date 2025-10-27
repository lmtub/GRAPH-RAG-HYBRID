static int find_pte32(CPUPPCState *env, struct mmu_ctx_hash32 *ctx,

                      target_ulong eaddr, int h, int rwx, int target_page_bits)

{

    hwaddr pteg_off;

    target_ulong pte0, pte1;

    int i, good = -1;

    int ret, r;



    ret = -1; /* No entry found */

    pteg_off = get_pteg_offset32(env, ctx->hash[h]);

    for (i = 0; i < HPTES_PER_GROUP; i++) {

        pte0 = ppc_hash32_load_hpte0(env, pteg_off + i*HASH_PTE_SIZE_32);

        pte1 = ppc_hash32_load_hpte1(env, pteg_off + i*HASH_PTE_SIZE_32);

        r = pte_check_hash32(ctx, pte0, pte1, h, rwx);

        LOG_MMU("Load pte from %08" HWADDR_PRIx " => " TARGET_FMT_lx " "

                TARGET_FMT_lx " %d %d %d " TARGET_FMT_lx "\n",

                pteg_off + (i * 8), pte0, pte1, (int)(pte0 >> 31), h,

                (int)((pte0 >> 6) & 1), ctx->ptem);

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

        if (ppc_hash32_pte_update_flags(ctx, &pte1, ret, rwx) == 1) {

            ppc_hash32_store_hpte1(env, pteg_off + good * HASH_PTE_SIZE_32,

                                   pte1);

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
