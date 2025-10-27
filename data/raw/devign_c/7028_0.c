static int mmu_translate_pte(CPUS390XState *env, target_ulong vaddr,

                             uint64_t asc, uint64_t asce,

                             target_ulong *raddr, int *flags, int rw)

{

    if (asce & _PAGE_INVALID) {

        DPRINTF("%s: PTE=0x%" PRIx64 " invalid\n", __func__, asce);

        trigger_page_fault(env, vaddr, PGM_PAGE_TRANS, asc, rw);

        return -1;

    }



    if (asce & _PAGE_RO) {

        *flags &= ~PAGE_WRITE;

    }



    *raddr = asce & _ASCE_ORIGIN;



    PTE_DPRINTF("%s: PTE=0x%" PRIx64 "\n", __func__, asce);



    return 0;

}
