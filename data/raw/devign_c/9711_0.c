static void tlb_unprotect_code_phys(CPUState *env, ram_addr_t ram_addr,

                                    target_ulong vaddr)

{

    phys_ram_dirty[ram_addr >> TARGET_PAGE_BITS] |= CODE_DIRTY_FLAG;

}
