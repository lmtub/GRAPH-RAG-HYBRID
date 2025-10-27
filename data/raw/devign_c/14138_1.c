static int mmu_translate_asc(CPUS390XState *env, target_ulong vaddr,

                             uint64_t asc, target_ulong *raddr, int *flags,

                             int rw, bool exc)

{

    uint64_t asce = 0;

    int level;

    int r;



    switch (asc) {

    case PSW_ASC_PRIMARY:

        PTE_DPRINTF("%s: asc=primary\n", __func__);

        asce = env->cregs[1];

        break;

    case PSW_ASC_SECONDARY:

        PTE_DPRINTF("%s: asc=secondary\n", __func__);

        asce = env->cregs[7];

        break;

    case PSW_ASC_HOME:

        PTE_DPRINTF("%s: asc=home\n", __func__);

        asce = env->cregs[13];

        break;

    }



    if (asce & _ASCE_REAL_SPACE) {

        /* direct mapping */

        *raddr = vaddr;

        return 0;

    }



    level = asce & _ASCE_TYPE_MASK;

    switch (level) {

    case _ASCE_TYPE_REGION1:

        if ((vaddr >> 62) > (asce & _ASCE_TABLE_LENGTH)) {

            trigger_page_fault(env, vaddr, PGM_REG_FIRST_TRANS, asc, rw, exc);

            return -1;

        }

        break;

    case _ASCE_TYPE_REGION2:

        if (vaddr & 0xffe0000000000000ULL) {

            DPRINTF("%s: vaddr doesn't fit 0x%16" PRIx64

                    " 0xffe0000000000000ULL\n", __func__, vaddr);

            trigger_page_fault(env, vaddr, PGM_TRANS_SPEC, asc, rw, exc);

            return -1;

        }

        if ((vaddr >> 51 & 3) > (asce & _ASCE_TABLE_LENGTH)) {

            trigger_page_fault(env, vaddr, PGM_REG_SEC_TRANS, asc, rw, exc);

            return -1;

        }

        break;

    case _ASCE_TYPE_REGION3:

        if (vaddr & 0xfffffc0000000000ULL) {

            DPRINTF("%s: vaddr doesn't fit 0x%16" PRIx64

                    " 0xfffffc0000000000ULL\n", __func__, vaddr);

            trigger_page_fault(env, vaddr, PGM_TRANS_SPEC, asc, rw, exc);

            return -1;

        }

        if ((vaddr >> 40 & 3) > (asce & _ASCE_TABLE_LENGTH)) {

            trigger_page_fault(env, vaddr, PGM_REG_THIRD_TRANS, asc, rw, exc);

            return -1;

        }

        break;

    case _ASCE_TYPE_SEGMENT:

        if (vaddr & 0xffffffff80000000ULL) {

            DPRINTF("%s: vaddr doesn't fit 0x%16" PRIx64

                    " 0xffffffff80000000ULL\n", __func__, vaddr);

            trigger_page_fault(env, vaddr, PGM_TRANS_SPEC, asc, rw, exc);

            return -1;

        }

        if ((vaddr >> 29 & 3) > (asce & _ASCE_TABLE_LENGTH)) {

            trigger_page_fault(env, vaddr, PGM_SEGMENT_TRANS, asc, rw, exc);

            return -1;

        }

        break;

    }



    r = mmu_translate_region(env, vaddr, asc, asce, level, raddr, flags, rw,

                             exc);

    if ((rw == 1) && !(*flags & PAGE_WRITE)) {

        trigger_prot_fault(env, vaddr, asc, rw, exc);

        return -1;

    }



    return r;

}
