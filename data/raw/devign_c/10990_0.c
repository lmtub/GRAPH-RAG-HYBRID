int r4k_map_address (CPUMIPSState *env, hwaddr *physical, int *prot,

                     target_ulong address, int rw, int access_type)

{

    uint8_t ASID = env->CP0_EntryHi & 0xFF;

    int i;



    for (i = 0; i < env->tlb->tlb_in_use; i++) {

        r4k_tlb_t *tlb = &env->tlb->mmu.r4k.tlb[i];

        /* 1k pages are not supported. */

        target_ulong mask = tlb->PageMask | ~(TARGET_PAGE_MASK << 1);

        target_ulong tag = address & ~mask;

        target_ulong VPN = tlb->VPN & ~mask;

#if defined(TARGET_MIPS64)

        tag &= env->SEGMask;

#endif



        /* Check ASID, virtual page number & size */

        if ((tlb->G == 1 || tlb->ASID == ASID) && VPN == tag) {

            /* TLB match */

            int n = !!(address & mask & ~(mask >> 1));

            /* Check access rights */

            if (!(n ? tlb->V1 : tlb->V0)) {

                return TLBRET_INVALID;

            }

            if (rw == MMU_INST_FETCH && (n ? tlb->XI1 : tlb->XI0)) {

                return TLBRET_XI;

            }

            if (rw == MMU_DATA_LOAD && (n ? tlb->RI1 : tlb->RI0)) {

                return TLBRET_RI;

            }

            if (rw != MMU_DATA_STORE || (n ? tlb->D1 : tlb->D0)) {

                *physical = tlb->PFN[n] | (address & (mask >> 1));

                *prot = PAGE_READ;

                if (n ? tlb->D1 : tlb->D0)

                    *prot |= PAGE_WRITE;

                return TLBRET_MATCH;

            }

            return TLBRET_DIRTY;

        }

    }

    return TLBRET_NOMATCH;

}
