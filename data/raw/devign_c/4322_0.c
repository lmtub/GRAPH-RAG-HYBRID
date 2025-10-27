static int ppc_hash64_translate(CPUPPCState *env, struct mmu_ctx_hash64 *ctx,

                                target_ulong eaddr, int rwx)

{

    int ret;

    ppc_slb_t *slb;

    hwaddr pte_offset;

    ppc_hash_pte64_t pte;

    int target_page_bits;



    assert((rwx == 0) || (rwx == 1) || (rwx == 2));



    /* 1. Handle real mode accesses */

    if (((rwx == 2) && (msr_ir == 0)) || ((rwx != 2) && (msr_dr == 0))) {

        /* Translation is off */

        /* In real mode the top 4 effective address bits are ignored */

        ctx->raddr = eaddr & 0x0FFFFFFFFFFFFFFFULL;

        ctx->prot = PAGE_READ | PAGE_EXEC | PAGE_WRITE;

        return 0;

    }



    /* 2. Translation is on, so look up the SLB */

    slb = slb_lookup(env, eaddr);



    if (!slb) {

        return -5;

    }



    /* 3. Check for segment level no-execute violation */

    if ((rwx == 2) && (slb->vsid & SLB_VSID_N)) {

        return -3;

    }



    /* 4. Locate the PTE in the hash table */

    pte_offset = ppc_hash64_htab_lookup(env, slb, eaddr, &pte);

    if (pte_offset == -1) {

        return -1;

    }

    LOG_MMU("found PTE at offset %08" HWADDR_PRIx "\n", pte_offset);



    /* 5. Check access permissions */

    ctx->key = !!(msr_pr ? (slb->vsid & SLB_VSID_KP)

                  : (slb->vsid & SLB_VSID_KS));





    int access, pp;

    bool nx;



    pp = (pte.pte1 & HPTE64_R_PP) | ((pte.pte1 & HPTE64_R_PP0) >> 61);

    /* No execute if either noexec or guarded bits set */

    nx = (pte.pte1 & HPTE64_R_N) || (pte.pte1 & HPTE64_R_G);

    /* Compute access rights */

    access = ppc_hash64_pp_check(ctx->key, pp, nx);

    /* Keep the matching PTE informations */

    ctx->raddr = pte.pte1;

    ctx->prot = access;

    ret = ppc_hash64_check_prot(ctx->prot, rwx);



    if (ret) {

        /* Access right violation */

        LOG_MMU("PTE access rejected\n");

        return ret;

    }



    LOG_MMU("PTE access granted !\n");



    /* 6. Update PTE referenced and changed bits if necessary */



    if (ppc_hash64_pte_update_flags(ctx, &pte.pte1, ret, rwx) == 1) {

        ppc_hash64_store_hpte1(env, pte_offset, pte.pte1);

    }



    /* We have a TLB that saves 4K pages, so let's

     * split a huge page to 4k chunks */

    target_page_bits = (slb->vsid & SLB_VSID_L)

        ? TARGET_PAGE_BITS_16M : TARGET_PAGE_BITS;

    if (target_page_bits != TARGET_PAGE_BITS) {

        ctx->raddr |= (eaddr & ((1 << target_page_bits) - 1))

                      & TARGET_PAGE_MASK;

    }

    return ret;

}
