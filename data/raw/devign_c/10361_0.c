static int mmu_translate_asce(CPUS390XState *env, target_ulong vaddr,

                              uint64_t asc, uint64_t asce, int level,

                              target_ulong *raddr, int *flags, int rw)

{

    CPUState *cs = CPU(s390_env_get_cpu(env));

    uint64_t offs = 0;

    uint64_t origin;

    uint64_t new_asce;



    PTE_DPRINTF("%s: 0x%" PRIx64 "\n", __func__, asce);



    if (((level != _ASCE_TYPE_SEGMENT) && (asce & _REGION_ENTRY_INV)) ||

        ((level == _ASCE_TYPE_SEGMENT) && (asce & _SEGMENT_ENTRY_INV))) {

        /* XXX different regions have different faults */

        DPRINTF("%s: invalid region\n", __func__);

        trigger_page_fault(env, vaddr, PGM_SEGMENT_TRANS, asc, rw);

        return -1;

    }



    if ((level <= _ASCE_TYPE_MASK) && ((asce & _ASCE_TYPE_MASK) != level)) {

        trigger_page_fault(env, vaddr, PGM_TRANS_SPEC, asc, rw);

        return -1;

    }



    if (asce & _ASCE_REAL_SPACE) {

        /* direct mapping */



        *raddr = vaddr;

        return 0;

    }



    origin = asce & _ASCE_ORIGIN;



    switch (level) {

    case _ASCE_TYPE_REGION1 + 4:

        offs = (vaddr >> 50) & 0x3ff8;

        break;

    case _ASCE_TYPE_REGION1:

        offs = (vaddr >> 39) & 0x3ff8;

        break;

    case _ASCE_TYPE_REGION2:

        offs = (vaddr >> 28) & 0x3ff8;

        break;

    case _ASCE_TYPE_REGION3:

        offs = (vaddr >> 17) & 0x3ff8;

        break;

    case _ASCE_TYPE_SEGMENT:

        offs = (vaddr >> 9) & 0x07f8;

        origin = asce & _SEGMENT_ENTRY_ORIGIN;

        break;

    }



    /* XXX region protection flags */

    /* *flags &= ~PAGE_WRITE */



    new_asce = ldq_phys(cs->as, origin + offs);

    PTE_DPRINTF("%s: 0x%" PRIx64 " + 0x%" PRIx64 " => 0x%016" PRIx64 "\n",

                __func__, origin, offs, new_asce);



    if (level == _ASCE_TYPE_SEGMENT) {

        /* 4KB page */

        return mmu_translate_pte(env, vaddr, asc, new_asce, raddr, flags, rw);

    } else if (level - 4 == _ASCE_TYPE_SEGMENT &&

               (new_asce & _SEGMENT_ENTRY_FC) && (env->cregs[0] & CR0_EDAT)) {

        /* 1MB page */

        return mmu_translate_sfaa(env, vaddr, asc, new_asce, raddr, flags, rw);

    } else {

        /* yet another region */

        return mmu_translate_asce(env, vaddr, asc, new_asce, level - 4, raddr,

                                  flags, rw);

    }

}
