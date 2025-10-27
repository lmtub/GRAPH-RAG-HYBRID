static int mmu_translate_region(CPUS390XState *env, target_ulong vaddr,

                                uint64_t asc, uint64_t entry, int level,

                                target_ulong *raddr, int *flags, int rw,

                                bool exc)

{

    CPUState *cs = CPU(s390_env_get_cpu(env));

    uint64_t origin, offs, new_entry;

    const int pchks[4] = {

        PGM_SEGMENT_TRANS, PGM_REG_THIRD_TRANS,

        PGM_REG_SEC_TRANS, PGM_REG_FIRST_TRANS

    };



    PTE_DPRINTF("%s: 0x%" PRIx64 "\n", __func__, entry);



    origin = entry & _REGION_ENTRY_ORIGIN;

    offs = (vaddr >> (17 + 11 * level / 4)) & 0x3ff8;



    new_entry = ldq_phys(cs->as, origin + offs);

    PTE_DPRINTF("%s: 0x%" PRIx64 " + 0x%" PRIx64 " => 0x%016" PRIx64 "\n",

                __func__, origin, offs, new_entry);



    if ((new_entry & _REGION_ENTRY_INV) != 0) {

        DPRINTF("%s: invalid region\n", __func__);

        trigger_page_fault(env, vaddr, pchks[level / 4], asc, rw, exc);

        return -1;

    }



    if ((new_entry & _REGION_ENTRY_TYPE_MASK) != level) {

        trigger_page_fault(env, vaddr, PGM_TRANS_SPEC, asc, rw, exc);

        return -1;

    }



    /* XXX region protection flags */

    /* *flags &= ~PAGE_WRITE */



    if (level == _ASCE_TYPE_SEGMENT) {

        return mmu_translate_segment(env, vaddr, asc, new_entry, raddr, flags,

                                     rw, exc);

    }



    /* Check region table offset and length */

    offs = (vaddr >> (28 + 11 * (level - 4) / 4)) & 3;

    if (offs < ((new_entry & _REGION_ENTRY_TF) >> 6)

        || offs > (new_entry & _REGION_ENTRY_LENGTH)) {

        DPRINTF("%s: invalid offset or len (%lx)\n", __func__, new_entry);

        trigger_page_fault(env, vaddr, pchks[level / 4 - 1], asc, rw, exc);

        return -1;

    }



    /* yet another region */

    return mmu_translate_region(env, vaddr, asc, new_entry, level - 4,

                                raddr, flags, rw, exc);

}
