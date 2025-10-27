static inline int get_segment(CPUState *env, mmu_ctx_t *ctx,

                              target_ulong eaddr, int rw, int type)

{

    target_phys_addr_t hash;

    target_ulong vsid;

    int ds, pr, target_page_bits;

    int ret, ret2;



    pr = msr_pr;

    ctx->eaddr = eaddr;

#if defined(TARGET_PPC64)

    if (env->mmu_model & POWERPC_MMU_64) {

        ppc_slb_t *slb;

        target_ulong pageaddr;

        int segment_bits;



        LOG_MMU("Check SLBs\n");

        slb = slb_lookup(env, eaddr);

        if (!slb) {

            return -5;

        }



        if (slb->vsid & SLB_VSID_B) {

            vsid = (slb->vsid & SLB_VSID_VSID) >> SLB_VSID_SHIFT_1T;

            segment_bits = 40;

        } else {

            vsid = (slb->vsid & SLB_VSID_VSID) >> SLB_VSID_SHIFT;

            segment_bits = 28;

        }



        target_page_bits = (slb->vsid & SLB_VSID_L)

            ? TARGET_PAGE_BITS_16M : TARGET_PAGE_BITS;

        ctx->key = !!(pr ? (slb->vsid & SLB_VSID_KP)

                      : (slb->vsid & SLB_VSID_KS));

        ds = 0;

        ctx->nx = !!(slb->vsid & SLB_VSID_N);



        pageaddr = eaddr & ((1ULL << segment_bits)

                            - (1ULL << target_page_bits));

        if (slb->vsid & SLB_VSID_B) {

            hash = vsid ^ (vsid << 25) ^ (pageaddr >> target_page_bits);

        } else {

            hash = vsid ^ (pageaddr >> target_page_bits);

        }

        /* Only 5 bits of the page index are used in the AVPN */

        ctx->ptem = (slb->vsid & SLB_VSID_PTEM) |

            ((pageaddr >> 16) & ((1ULL << segment_bits) - 0x80));

    } else

#endif /* defined(TARGET_PPC64) */

    {

        target_ulong sr, pgidx;



        sr = env->sr[eaddr >> 28];

        ctx->key = (((sr & 0x20000000) && (pr != 0)) ||

                    ((sr & 0x40000000) && (pr == 0))) ? 1 : 0;

        ds = sr & 0x80000000 ? 1 : 0;

        ctx->nx = sr & 0x10000000 ? 1 : 0;

        vsid = sr & 0x00FFFFFF;

        target_page_bits = TARGET_PAGE_BITS;

        LOG_MMU("Check segment v=" TARGET_FMT_lx " %d " TARGET_FMT_lx " nip="

                TARGET_FMT_lx " lr=" TARGET_FMT_lx

                " ir=%d dr=%d pr=%d %d t=%d\n",

                eaddr, (int)(eaddr >> 28), sr, env->nip, env->lr, (int)msr_ir,

                (int)msr_dr, pr != 0 ? 1 : 0, rw, type);

        pgidx = (eaddr & ~SEGMENT_MASK_256M) >> target_page_bits;

        hash = vsid ^ pgidx;

        ctx->ptem = (vsid << 7) | (pgidx >> 10);

    }

    LOG_MMU("pte segment: key=%d ds %d nx %d vsid " TARGET_FMT_lx "\n",

            ctx->key, ds, ctx->nx, vsid);

    ret = -1;

    if (!ds) {

        /* Check if instruction fetch is allowed, if needed */

        if (type != ACCESS_CODE || ctx->nx == 0) {

            /* Page address translation */

            LOG_MMU("htab_base " TARGET_FMT_plx " htab_mask " TARGET_FMT_plx

                    " hash " TARGET_FMT_plx "\n",

                    env->htab_base, env->htab_mask, hash);

            ctx->hash[0] = hash;

            ctx->hash[1] = ~hash;



            /* Initialize real address with an invalid value */

            ctx->raddr = (target_phys_addr_t)-1ULL;

            if (unlikely(env->mmu_model == POWERPC_MMU_SOFT_6xx ||

                         env->mmu_model == POWERPC_MMU_SOFT_74xx)) {

                /* Software TLB search */

                ret = ppc6xx_tlb_check(env, ctx, eaddr, rw, type);

            } else {

                LOG_MMU("0 htab=" TARGET_FMT_plx "/" TARGET_FMT_plx

                        " vsid=" TARGET_FMT_lx " ptem=" TARGET_FMT_lx

                        " hash=" TARGET_FMT_plx "\n",

                        env->htab_base, env->htab_mask, vsid, ctx->ptem,

                        ctx->hash[0]);

                /* Primary table lookup */

                ret = find_pte(env, ctx, 0, rw, type, target_page_bits);

                if (ret < 0) {

                    /* Secondary table lookup */

                    if (eaddr != 0xEFFFFFFF)

                        LOG_MMU("1 htab=" TARGET_FMT_plx "/" TARGET_FMT_plx

                                " vsid=" TARGET_FMT_lx " api=" TARGET_FMT_lx

                                " hash=" TARGET_FMT_plx " pg_addr="

                                TARGET_FMT_plx "\n", env->htab_base,

                                env->htab_mask, vsid, ctx->ptem, ctx->hash[1]);

                    ret2 = find_pte(env, ctx, 1, rw, type,

                                    target_page_bits);

                    if (ret2 != -1)

                        ret = ret2;

                }

            }

#if defined (DUMP_PAGE_TABLES)

            if (qemu_log_enabled()) {

                target_phys_addr_t curaddr;

                uint32_t a0, a1, a2, a3;

                qemu_log("Page table: " TARGET_FMT_plx " len " TARGET_FMT_plx

                         "\n", sdr, mask + 0x80);

                for (curaddr = sdr; curaddr < (sdr + mask + 0x80);

                     curaddr += 16) {

                    a0 = ldl_phys(curaddr);

                    a1 = ldl_phys(curaddr + 4);

                    a2 = ldl_phys(curaddr + 8);

                    a3 = ldl_phys(curaddr + 12);

                    if (a0 != 0 || a1 != 0 || a2 != 0 || a3 != 0) {

                        qemu_log(TARGET_FMT_plx ": %08x %08x %08x %08x\n",

                                 curaddr, a0, a1, a2, a3);

                    }

                }

            }

#endif

        } else {

            LOG_MMU("No access allowed\n");

            ret = -3;

        }

    } else {

        LOG_MMU("direct store...\n");

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

            qemu_log("ERROR: instruction should not need "

                        "address translation\n");

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
