void helper_booke206_tlbwe(CPUPPCState *env)

{

    PowerPCCPU *cpu = ppc_env_get_cpu(env);

    uint32_t tlbncfg, tlbn;

    ppcmas_tlb_t *tlb;

    uint32_t size_tlb, size_ps;

    target_ulong mask;





    switch (env->spr[SPR_BOOKE_MAS0] & MAS0_WQ_MASK) {

    case MAS0_WQ_ALWAYS:

        /* good to go, write that entry */

        break;

    case MAS0_WQ_COND:

        /* XXX check if reserved */

        if (0) {

            return;

        }

        break;

    case MAS0_WQ_CLR_RSRV:

        /* XXX clear entry */

        return;

    default:

        /* no idea what to do */

        return;

    }



    if (((env->spr[SPR_BOOKE_MAS0] & MAS0_ATSEL) == MAS0_ATSEL_LRAT) &&

        !msr_gs) {

        /* XXX we don't support direct LRAT setting yet */

        fprintf(stderr, "cpu: don't support LRAT setting yet\n");

        return;

    }



    tlbn = (env->spr[SPR_BOOKE_MAS0] & MAS0_TLBSEL_MASK) >> MAS0_TLBSEL_SHIFT;

    tlbncfg = env->spr[SPR_BOOKE_TLB0CFG + tlbn];



    tlb = booke206_cur_tlb(env);



    if (!tlb) {

        raise_exception_err_ra(env, POWERPC_EXCP_PROGRAM,

                               POWERPC_EXCP_INVAL |

                               POWERPC_EXCP_INVAL_INVAL, GETPC());

    }



    /* check that we support the targeted size */

    size_tlb = (env->spr[SPR_BOOKE_MAS1] & MAS1_TSIZE_MASK) >> MAS1_TSIZE_SHIFT;

    size_ps = booke206_tlbnps(env, tlbn);

    if ((env->spr[SPR_BOOKE_MAS1] & MAS1_VALID) && (tlbncfg & TLBnCFG_AVAIL) &&

        !(size_ps & (1 << size_tlb))) {

        raise_exception_err_ra(env, POWERPC_EXCP_PROGRAM,

                               POWERPC_EXCP_INVAL |

                               POWERPC_EXCP_INVAL_INVAL, GETPC());

    }



    if (msr_gs) {

        cpu_abort(CPU(cpu), "missing HV implementation\n");

    }

    tlb->mas7_3 = ((uint64_t)env->spr[SPR_BOOKE_MAS7] << 32) |

        env->spr[SPR_BOOKE_MAS3];

    tlb->mas1 = env->spr[SPR_BOOKE_MAS1];



    if ((env->spr[SPR_MMUCFG] & MMUCFG_MAVN) == MMUCFG_MAVN_V2) {

        /* For TLB which has a fixed size TSIZE is ignored with MAV2 */

        booke206_fixed_size_tlbn(env, tlbn, tlb);

    } else {

        if (!(tlbncfg & TLBnCFG_AVAIL)) {

            /* force !AVAIL TLB entries to correct page size */

            tlb->mas1 &= ~MAS1_TSIZE_MASK;

            /* XXX can be configured in MMUCSR0 */

            tlb->mas1 |= (tlbncfg & TLBnCFG_MINSIZE) >> 12;

        }

    }



    /* Make a mask from TLB size to discard invalid bits in EPN field */

    mask = ~(booke206_tlb_to_page_size(env, tlb) - 1);

    /* Add a mask for page attributes */

    mask |= MAS2_ACM | MAS2_VLE | MAS2_W | MAS2_I | MAS2_M | MAS2_G | MAS2_E;



    if (!msr_cm) {

        /* Executing a tlbwe instruction in 32-bit mode will set

         * bits 0:31 of the TLB EPN field to zero.

         */

        mask &= 0xffffffff;

    }



    tlb->mas2 = env->spr[SPR_BOOKE_MAS2] & mask;



    if (!(tlbncfg & TLBnCFG_IPROT)) {

        /* no IPROT supported by TLB */

        tlb->mas1 &= ~MAS1_IPROT;

    }



    if (booke206_tlb_to_page_size(env, tlb) == TARGET_PAGE_SIZE) {

        tlb_flush_page(CPU(cpu), tlb->mas2 & MAS2_EPN_MASK);

    } else {

        tlb_flush(CPU(cpu));

    }

}
