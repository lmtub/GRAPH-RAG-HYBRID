static inline int _find_pte(CPUState *env, mmu_ctx_t *ctx, int is_64b, int h,

                            int rw, int type, int target_page_bits)

{

    target_phys_addr_t pteg_off;

    target_ulong pte0, pte1;

    int i, good = -1;

    int ret, r;



    ret = -1; /* No entry found */

    pteg_off = get_pteg_offset(env, ctx->hash[h],

                               is_64b ? HASH_PTE_SIZE_64 : HASH_PTE_SIZE_32);

    for (i = 0; i < 8; i++) {

#if defined(TARGET_PPC64)

        if (is_64b) {

            if (env->external_htab) {

                pte0 = ldq_p(env->external_htab + pteg_off + (i * 16));

                pte1 = ldq_p(env->external_htab + pteg_off + (i * 16) + 8);

            } else {

                pte0 = ldq_phys(env->htab_base + pteg_off + (i * 16));

                pte1 = ldq_phys(env->htab_base + pteg_off + (i * 16) + 8);

            }



            /* We have a TLB that saves 4K pages, so let's

             * split a huge page to 4k chunks */

            if (target_page_bits != TARGET_PAGE_BITS)

                pte1 |= (ctx->eaddr & (( 1 << target_page_bits ) - 1))

                        & TARGET_PAGE_MASK;



            r = pte64_check(ctx, pte0, pte1, h, rw, type);

            LOG_MMU("Load pte from " TARGET_FMT_lx " => " TARGET_FMT_lx " "

                    TARGET_FMT_lx " %d %d %d " TARGET_FMT_lx "\n",

                    pteg_base + (i * 16), pte0, pte1, (int)(pte0 & 1), h,

                    (int)((pte0 >> 1) & 1), ctx->ptem);

        } else

#endif

        {

            if (env->external_htab) {

                pte0 = ldl_p(env->external_htab + pteg_off + (i * 8));

                pte1 = ldl_p(env->external_htab + pteg_off + (i * 8) + 4);

            } else {

                pte0 = ldl_phys(env->htab_base + pteg_off + (i * 8));

                pte1 = ldl_phys(env->htab_base + pteg_off + (i * 8) + 4);

            }

            r = pte32_check(ctx, pte0, pte1, h, rw, type);

            LOG_MMU("Load pte from " TARGET_FMT_lx " => " TARGET_FMT_lx " "

                    TARGET_FMT_lx " %d %d %d " TARGET_FMT_lx "\n",

                    pteg_base + (i * 8), pte0, pte1, (int)(pte0 >> 31), h,

                    (int)((pte0 >> 6) & 1), ctx->ptem);

        }

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

        LOG_MMU("found PTE at addr " TARGET_FMT_lx " prot=%01x ret=%d\n",

                ctx->raddr, ctx->prot, ret);

        /* Update page flags */

        pte1 = ctx->raddr;

        if (pte_update_flags(ctx, &pte1, ret, rw) == 1) {

#if defined(TARGET_PPC64)

            if (is_64b) {

                if (env->external_htab) {

                    stq_p(env->external_htab + pteg_off + (good * 16) + 8,

                          pte1);

                } else {

                    stq_phys_notdirty(env->htab_base + pteg_off +

                                      (good * 16) + 8, pte1);

                }

            } else

#endif

            {

                if (env->external_htab) {

                    stl_p(env->external_htab + pteg_off + (good * 8) + 4,

                          pte1);

                } else {

                    stl_phys_notdirty(env->htab_base + pteg_off +

                                      (good * 8) + 4, pte1);

                }

            }

        }

    }



    return ret;

}
