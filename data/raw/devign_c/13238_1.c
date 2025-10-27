static always_inline int get_segment (CPUState *env, mmu_ctx_t *ctx,

                                      target_ulong eaddr, int rw, int type)

{

    target_phys_addr_t sdr, hash, mask, sdr_mask, htab_mask;

    target_ulong sr, vsid, vsid_mask, pgidx, page_mask;

#if defined(TARGET_PPC64)

    int attr;

#endif

    int ds, vsid_sh, sdr_sh, pr;

    int ret, ret2;



    pr = msr_pr;

#if defined(TARGET_PPC64)

    if (env->mmu_model == POWERPC_MMU_64B) {

#if defined (DEBUG_MMU)

        if (loglevel != 0) {

            fprintf(logfile, "Check SLBs\n");

        }

#endif

        ret = slb_lookup(env, eaddr, &vsid, &page_mask, &attr);

        if (ret < 0)

            return ret;

        ctx->key = ((attr & 0x40) && (pr != 0)) ||

            ((attr & 0x80) && (pr == 0)) ? 1 : 0;

        ds = 0;

        ctx->nx = attr & 0x20 ? 1 : 0;

        vsid_mask = 0x00003FFFFFFFFF80ULL;

        vsid_sh = 7;

        sdr_sh = 18;

        sdr_mask = 0x3FF80;

    } else

#endif /* defined(TARGET_PPC64) */

    {

        sr = env->sr[eaddr >> 28];

        page_mask = 0x0FFFFFFF;

        ctx->key = (((sr & 0x20000000) && (pr != 0)) ||

                    ((sr & 0x40000000) && (pr == 0))) ? 1 : 0;

        ds = sr & 0x80000000 ? 1 : 0;

        ctx->nx = sr & 0x10000000 ? 1 : 0;

        vsid = sr & 0x00FFFFFF;

        vsid_mask = 0x01FFFFC0;

        vsid_sh = 6;

        sdr_sh = 16;

        sdr_mask = 0xFFC0;

#if defined (DEBUG_MMU)

        if (loglevel != 0) {

            fprintf(logfile, "Check segment v=0x" ADDRX " %d 0x" ADDRX

                    " nip=0x" ADDRX " lr=0x" ADDRX

                    " ir=%d dr=%d pr=%d %d t=%d\n",

                    eaddr, (int)(eaddr >> 28), sr, env->nip,

                    env->lr, (int)msr_ir, (int)msr_dr, pr != 0 ? 1 : 0,

                    rw, type);

        }

#endif

    }

#if defined (DEBUG_MMU)

    if (loglevel != 0) {

        fprintf(logfile, "pte segment: key=%d ds %d nx %d vsid " ADDRX "\n",

                ctx->key, ds, ctx->nx, vsid);

    }

#endif

    ret = -1;

    if (!ds) {

        /* Check if instruction fetch is allowed, if needed */

        if (type != ACCESS_CODE || ctx->nx == 0) {

            /* Page address translation */

            /* Primary table address */

            sdr = env->sdr1;

            pgidx = (eaddr & page_mask) >> TARGET_PAGE_BITS;

#if defined(TARGET_PPC64)

            if (env->mmu_model == POWERPC_MMU_64B) {

                htab_mask = 0x0FFFFFFF >> (28 - (sdr & 0x1F));

                /* XXX: this is false for 1 TB segments */

                hash = ((vsid ^ pgidx) << vsid_sh) & vsid_mask;

            } else

#endif

            {

                htab_mask = sdr & 0x000001FF;

                hash = ((vsid ^ pgidx) << vsid_sh) & vsid_mask;

            }

            mask = (htab_mask << sdr_sh) | sdr_mask;

#if defined (DEBUG_MMU)

            if (loglevel != 0) {

                fprintf(logfile, "sdr " PADDRX " sh %d hash " PADDRX " mask "

                        PADDRX " " ADDRX "\n", sdr, sdr_sh, hash, mask,

                        page_mask);

            }

#endif

            ctx->pg_addr[0] = get_pgaddr(sdr, sdr_sh, hash, mask);

            /* Secondary table address */

            hash = (~hash) & vsid_mask;

#if defined (DEBUG_MMU)

            if (loglevel != 0) {

                fprintf(logfile, "sdr " PADDRX " sh %d hash " PADDRX " mask "

                        PADDRX "\n", sdr, sdr_sh, hash, mask);

            }

#endif

            ctx->pg_addr[1] = get_pgaddr(sdr, sdr_sh, hash, mask);

#if defined(TARGET_PPC64)

            if (env->mmu_model == POWERPC_MMU_64B) {

                /* Only 5 bits of the page index are used in the AVPN */

                ctx->ptem = (vsid << 12) | ((pgidx >> 4) & 0x0F80);

            } else

#endif

            {

                ctx->ptem = (vsid << 7) | (pgidx >> 10);

            }

            /* Initialize real address with an invalid value */

            ctx->raddr = (target_ulong)-1;

            if (unlikely(env->mmu_model == POWERPC_MMU_SOFT_6xx ||

                         env->mmu_model == POWERPC_MMU_SOFT_74xx)) {

                /* Software TLB search */

                ret = ppc6xx_tlb_check(env, ctx, eaddr, rw, type);

            } else {

#if defined (DEBUG_MMU)

                if (loglevel != 0) {

                    fprintf(logfile, "0 sdr1=0x" PADDRX " vsid=0x%06x "

                            "api=0x%04x hash=0x%07x pg_addr=0x" PADDRX "\n",

                            sdr, (uint32_t)vsid, (uint32_t)pgidx,

                            (uint32_t)hash, ctx->pg_addr[0]);

                }

#endif

                /* Primary table lookup */

                ret = find_pte(env, ctx, 0, rw, type);

                if (ret < 0) {

                    /* Secondary table lookup */

#if defined (DEBUG_MMU)

                    if (eaddr != 0xEFFFFFFF && loglevel != 0) {

                        fprintf(logfile,

                                "1 sdr1=0x" PADDRX " vsid=0x%06x api=0x%04x "

                                "hash=0x%05x pg_addr=0x" PADDRX "\n",

                                sdr, (uint32_t)vsid, (uint32_t)pgidx,

                                (uint32_t)hash, ctx->pg_addr[1]);

                    }

#endif

                    ret2 = find_pte(env, ctx, 1, rw, type);

                    if (ret2 != -1)

                        ret = ret2;

                }

            }

#if defined (DUMP_PAGE_TABLES)

            if (loglevel != 0) {

                target_phys_addr_t curaddr;

                uint32_t a0, a1, a2, a3;

                fprintf(logfile,

                        "Page table: " PADDRX " len " PADDRX "\n",

                        sdr, mask + 0x80);

                for (curaddr = sdr; curaddr < (sdr + mask + 0x80);

                     curaddr += 16) {

                    a0 = ldl_phys(curaddr);

                    a1 = ldl_phys(curaddr + 4);

                    a2 = ldl_phys(curaddr + 8);

                    a3 = ldl_phys(curaddr + 12);

                    if (a0 != 0 || a1 != 0 || a2 != 0 || a3 != 0) {

                        fprintf(logfile,

                                PADDRX ": %08x %08x %08x %08x\n",

                                curaddr, a0, a1, a2, a3);

                    }

                }

            }

#endif

        } else {

#if defined (DEBUG_MMU)

            if (loglevel != 0)

                fprintf(logfile, "No access allowed\n");

#endif

            ret = -3;

        }

    } else {

#if defined (DEBUG_MMU)

        if (loglevel != 0)

            fprintf(logfile, "direct store...\n");

#endif

        /* Direct-store segment : absolutely *BUGGY* for now */

        switch (type) {

        case ACCESS_INT:

            /* Integer load/store : only access allowed */

            break;

        case ACCESS_CODE:

            /* No code fetch is allowed in direct-store areas */

            return -4;

        case ACCESS_FLOAT:

            /* Floating point load/store */

            return -4;

        case ACCESS_RES:

            /* lwarx, ldarx or srwcx. */

            return -4;

        case ACCESS_CACHE:

            /* dcba, dcbt, dcbtst, dcbf, dcbi, dcbst, dcbz, or icbi */

            /* Should make the instruction do no-op.

             * As it already do no-op, it's quite easy :-)

             */

            ctx->raddr = eaddr;

            return 0;

        case ACCESS_EXT:

            /* eciwx or ecowx */

            return -4;

        default:

            if (logfile) {

                fprintf(logfile, "ERROR: instruction should not need "

                        "address translation\n");

            }

            return -4;

        }

        if ((rw == 1 || ctx->key != 1) && (rw == 0 || ctx->key != 0)) {

            ctx->raddr = eaddr;

            ret = 2;

        } else {

            ret = -2;

        }

    }



    return ret;

}
