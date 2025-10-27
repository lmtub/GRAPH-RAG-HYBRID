void tlb_fill(target_ulong addr, int is_write, int mmu_idx, void *retaddr)

{

    tlb_set_page(cpu_single_env,

            addr & ~(TARGET_PAGE_SIZE - 1),

            addr & ~(TARGET_PAGE_SIZE - 1),

            PAGE_READ | PAGE_WRITE | PAGE_EXEC,

            mmu_idx, TARGET_PAGE_SIZE);

}
