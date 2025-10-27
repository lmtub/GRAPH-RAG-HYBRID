static int find_pte32(CPUPPCState *env, struct mmu_ctx_hash32 *ctx,

                      target_ulong sr, target_ulong eaddr, int rwx)

{

    hwaddr pteg_off, pte_offset;

    ppc_hash_pte32_t pte;

    hwaddr hash;

    uint32_t vsid, pgidx, ptem;

    int ret;



    ret = -1; /* No entry found */

    vsid = sr & SR32_VSID;

    ctx->key = (((sr & SR32_KP) && (msr_pr != 0)) ||

                ((sr & SR32_KS) && (msr_pr == 0))) ? 1 : 0;

    pgidx = (eaddr & ~SEGMENT_MASK_256M) >> TARGET_PAGE_BITS;

    hash = vsid ^ pgidx;

    ptem = (vsid << 7) | (pgidx >> 10);



    /* Page address translation */

    LOG_MMU("htab_base " TARGET_FMT_plx " htab_mask " TARGET_FMT_plx

            " hash " TARGET_FMT_plx "\n",

            env->htab_base, env->htab_mask, hash);



    /* Primary PTEG lookup */

    LOG_MMU("0 htab=" TARGET_FMT_plx "/" TARGET_FMT_plx

            " vsid=%" PRIx32 " ptem=%" PRIx32

            " hash=" TARGET_FMT_plx "\n",

            env->htab_base, env->htab_mask, vsid, ptem, hash);

    pteg_off = get_pteg_offset32(env, hash);

    pte_offset = ppc_hash32_pteg_search(env, pteg_off, 0, ptem, &pte);

    if (pte_offset == -1) {

        /* Secondary PTEG lookup */

        LOG_MMU("1 htab=" TARGET_FMT_plx "/" TARGET_FMT_plx

                " vsid=%" PRIx32 " api=%" PRIx32

                " hash=" TARGET_FMT_plx "\n", env->htab_base,

                env->htab_mask, vsid, ptem, ~hash);

        pteg_off = get_pteg_offset32(env, ~hash);

        pte_offset = ppc_hash32_pteg_search(env, pteg_off, 1, ptem, &pte);

    }



    if (pte_offset != -1) {

        ret = pte_check_hash32(ctx, pte.pte0, pte.pte1, rwx);

        LOG_MMU("found PTE at addr %08" HWADDR_PRIx " prot=%01x ret=%d\n",

                ctx->raddr, ctx->prot, ret);

        /* Update page flags */

        if (ppc_hash32_pte_update_flags(ctx, &pte.pte1, ret, rwx) == 1) {

            ppc_hash32_store_hpte1(env, pte_offset, pte.pte1);

        }

    }



    return ret;

}
