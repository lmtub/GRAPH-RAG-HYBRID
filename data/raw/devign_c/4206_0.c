static int find_pte64(CPUPPCState *env, struct mmu_ctx_hash64 *ctx,

                      ppc_slb_t *slb, target_ulong eaddr, int rwx)

{

    hwaddr pteg_off, pte_offset;

    ppc_hash_pte64_t pte;

    uint64_t vsid, pageaddr, ptem;

    hwaddr hash;

    int segment_bits, target_page_bits;

    int ret;



    ret = -1; /* No entry found */



    if (slb->vsid & SLB_VSID_B) {

        vsid = (slb->vsid & SLB_VSID_VSID) >> SLB_VSID_SHIFT_1T;

        segment_bits = 40;

    } else {

        vsid = (slb->vsid & SLB_VSID_VSID) >> SLB_VSID_SHIFT;

        segment_bits = 28;

    }



    target_page_bits = (slb->vsid & SLB_VSID_L)

        ? TARGET_PAGE_BITS_16M : TARGET_PAGE_BITS;

    ctx->key = !!(msr_pr ? (slb->vsid & SLB_VSID_KP)

                  : (slb->vsid & SLB_VSID_KS));



    pageaddr = eaddr & ((1ULL << segment_bits)

                            - (1ULL << target_page_bits));

    if (slb->vsid & SLB_VSID_B) {

        hash = vsid ^ (vsid << 25) ^ (pageaddr >> target_page_bits);

    } else {

        hash = vsid ^ (pageaddr >> target_page_bits);

    }

    /* Only 5 bits of the page index are used in the AVPN */

    ptem = (slb->vsid & SLB_VSID_PTEM) |

        ((pageaddr >> 16) & ((1ULL << segment_bits) - 0x80));



    ret = -1;



    /* Page address translation */

    LOG_MMU("htab_base " TARGET_FMT_plx " htab_mask " TARGET_FMT_plx

            " hash " TARGET_FMT_plx "\n",

            env->htab_base, env->htab_mask, hash);





    /* Primary PTEG lookup */

    LOG_MMU("0 htab=" TARGET_FMT_plx "/" TARGET_FMT_plx

            " vsid=" TARGET_FMT_lx " ptem=" TARGET_FMT_lx

            " hash=" TARGET_FMT_plx "\n",

            env->htab_base, env->htab_mask, vsid, ptem,  hash);

    pteg_off = (hash * HASH_PTEG_SIZE_64) & env->htab_mask;

    pte_offset = ppc_hash64_pteg_search(env, pteg_off, 0, ptem, &pte);

    if (pte_offset == -1) {

        /* Secondary PTEG lookup */

        LOG_MMU("1 htab=" TARGET_FMT_plx "/" TARGET_FMT_plx

                " vsid=" TARGET_FMT_lx " api=" TARGET_FMT_lx

                " hash=" TARGET_FMT_plx "\n", env->htab_base,

                env->htab_mask, vsid, ptem, ~hash);



        pteg_off = (~hash * HASH_PTEG_SIZE_64) & env->htab_mask;

        pte_offset = ppc_hash64_pteg_search(env, pteg_off, 1, ptem, &pte);

    }



    if (pte_offset != -1) {

        ret = pte64_check(ctx, pte.pte0, pte.pte1, rwx);

        LOG_MMU("found PTE at addr %08" HWADDR_PRIx " prot=%01x ret=%d\n",

                ctx->raddr, ctx->prot, ret);

        /* Update page flags */

        if (ppc_hash64_pte_update_flags(ctx, &pte.pte1, ret, rwx) == 1) {

            ppc_hash64_store_hpte1(env, pte_offset, pte.pte1);

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
