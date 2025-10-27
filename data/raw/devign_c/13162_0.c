static always_inline int _find_pte (mmu_ctx_t *ctx, int is_64b, int h, int rw)

{

    target_ulong base, pte0, pte1;

    int i, good = -1;

    int ret, r;



    ret = -1; /* No entry found */

    base = ctx->pg_addr[h];

    for (i = 0; i < 8; i++) {

#if defined(TARGET_PPC64)

        if (is_64b) {

            pte0 = ldq_phys(base + (i * 16));

            pte1 =  ldq_phys(base + (i * 16) + 8);

            r = pte64_check(ctx, pte0, pte1, h, rw);

#if defined (DEBUG_MMU)

            if (loglevel != 0) {

                fprintf(logfile, "Load pte from 0x" ADDRX " => 0x" ADDRX

                        " 0x" ADDRX " %d %d %d 0x" ADDRX "\n",

                        base + (i * 16), pte0, pte1,

                        (int)(pte0 & 1), h, (int)((pte0 >> 1) & 1),

                        ctx->ptem);

            }

#endif

        } else

#endif

        {

            pte0 = ldl_phys(base + (i * 8));

            pte1 =  ldl_phys(base + (i * 8) + 4);

            r = pte32_check(ctx, pte0, pte1, h, rw);

#if defined (DEBUG_MMU)

            if (loglevel != 0) {

                fprintf(logfile, "Load pte from 0x" ADDRX " => 0x" ADDRX

                        " 0x" ADDRX " %d %d %d 0x" ADDRX "\n",

                        base + (i * 8), pte0, pte1,

                        (int)(pte0 >> 31), h, (int)((pte0 >> 6) & 1),

                        ctx->ptem);

            }

#endif

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

#if defined (DEBUG_MMU)

        if (loglevel != 0) {

            fprintf(logfile, "found PTE at addr 0x" PADDRX " prot=0x%01x "

                    "ret=%d\n",

                    ctx->raddr, ctx->prot, ret);

        }

#endif

        /* Update page flags */

        pte1 = ctx->raddr;

        if (pte_update_flags(ctx, &pte1, ret, rw) == 1) {

#if defined(TARGET_PPC64)

            if (is_64b) {

                stq_phys_notdirty(base + (good * 16) + 8, pte1);

            } else

#endif

            {

                stl_phys_notdirty(base + (good * 8) + 4, pte1);

            }

        }

    }



    return ret;

}
