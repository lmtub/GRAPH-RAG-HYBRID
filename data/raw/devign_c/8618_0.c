static int mmu_translate_sfaa(CPUS390XState *env, target_ulong vaddr,

                              uint64_t asc, uint64_t asce, target_ulong *raddr,

                              int *flags, int rw)

{

    if (asce & _SEGMENT_ENTRY_INV) {

        DPRINTF("%s: SEG=0x%" PRIx64 " invalid\n", __func__, asce);

        trigger_page_fault(env, vaddr, PGM_SEGMENT_TRANS, asc, rw);

        return -1;

    }



    if (asce & _SEGMENT_ENTRY_RO) {

        *flags &= ~PAGE_WRITE;

    }



    *raddr = (asce & 0xfffffffffff00000ULL) | (vaddr & 0xfffff);



    PTE_DPRINTF("%s: SEG=0x%" PRIx64 "\n", __func__, asce);



    return 0;

}
