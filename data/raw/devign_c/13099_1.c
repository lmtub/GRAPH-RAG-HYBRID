static int ppc6xx_tlb_check (CPUState *env, mmu_ctx_t *ctx,

                             target_ulong eaddr, int rw, int access_type)

{

    ppc_tlb_t *tlb;

    int nr, best, way;

    int ret;

    

    best = -1;

    ret = -1; /* No TLB found */

    for (way = 0; way < env->nb_ways; way++) {

        nr = ppc6xx_tlb_getnum(env, eaddr, way,

                               access_type == ACCESS_CODE ? 1 : 0);

        tlb = &env->tlb[nr];

        /* This test "emulates" the PTE index match for hardware TLBs */

        if ((eaddr & TARGET_PAGE_MASK) != tlb->EPN) {

#if defined (DEBUG_SOFTWARE_TLB)

            if (loglevel != 0) {

                fprintf(logfile, "TLB %d/%d %s [%08x %08x] <> %08x\n",

                        nr, env->nb_tlb,

                        pte_is_valid(tlb->pte0) ? "valid" : "inval",

                        tlb->EPN, tlb->EPN + TARGET_PAGE_SIZE, eaddr);

            }

#endif

            continue;

        }

#if defined (DEBUG_SOFTWARE_TLB)

        if (loglevel != 0) {

            fprintf(logfile, "TLB %d/%d %s %08x <> %08x %08x %c %c\n",

                    nr, env->nb_tlb,

                    pte_is_valid(tlb->pte0) ? "valid" : "inval",

                    tlb->EPN, eaddr, tlb->pte1,

                    rw ? 'S' : 'L', access_type == ACCESS_CODE ? 'I' : 'D');

        }

#endif

        switch (pte_check(ctx, tlb->pte0, tlb->pte1, 0, rw)) {

        case -3:

            /* TLB inconsistency */

            return -1;

        case -2:

            /* Access violation */

            ret = -2;

            best = nr;

            break;

        case -1:

        default:

            /* No match */

            break;

        case 0:

            /* access granted */

            /* XXX: we should go on looping to check all TLBs consistency

             *      but we can speed-up the whole thing as the

             *      result would be undefined if TLBs are not consistent.

             */

            ret = 0;

            best = nr;

            goto done;

        }

    }

    if (best != -1) {

    done:

#if defined (DEBUG_SOFTWARE_TLB)

        if (loglevel > 0) {

            fprintf(logfile, "found TLB at addr 0x%08lx prot=0x%01x ret=%d\n",

                    ctx->raddr & TARGET_PAGE_MASK, ctx->prot, ret);

        }

#endif

        /* Update page flags */

        pte_update_flags(ctx, &env->tlb[best].pte1, ret, rw);

    }



    return ret;

}
