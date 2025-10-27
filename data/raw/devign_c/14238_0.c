static int mmu_translate_segment(CPUS390XState *env, target_ulong vaddr,

                                 uint64_t asc, uint64_t st_entry,

                                 target_ulong *raddr, int *flags, int rw)

{

    CPUState *cs = CPU(s390_env_get_cpu(env));

    uint64_t origin, offs, pt_entry;



    if (st_entry & _SEGMENT_ENTRY_RO) {

        *flags &= ~PAGE_WRITE;

    }



    if ((st_entry & _SEGMENT_ENTRY_FC) && (env->cregs[0] & CR0_EDAT)) {

        /* Decode EDAT1 segment frame absolute address (1MB page) */

        *raddr = (st_entry & 0xfffffffffff00000ULL) | (vaddr & 0xfffff);

        PTE_DPRINTF("%s: SEG=0x%" PRIx64 "\n", __func__, st_entry);

        return 0;

    }



    /* Look up 4KB page entry */

    origin = st_entry & _SEGMENT_ENTRY_ORIGIN;

    offs  = (vaddr & VADDR_PX) >> 9;

    pt_entry = ldq_phys(cs->as, origin + offs);

    PTE_DPRINTF("%s: 0x%" PRIx64 " + 0x%" PRIx64 " => 0x%016" PRIx64 "\n",

                __func__, origin, offs, pt_entry);

    return mmu_translate_pte(env, vaddr, asc, pt_entry, raddr, flags, rw);

}
