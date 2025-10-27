void tlb_reset_dirty_range(CPUTLBEntry *tlb_entry, uintptr_t start,

                           uintptr_t length)

{

    uintptr_t addr;



    if (tlb_is_dirty_ram(tlb_entry)) {

        addr = (tlb_entry->addr_write & TARGET_PAGE_MASK) + tlb_entry->addend;

        if ((addr - start) < length) {

            tlb_entry->addr_write |= TLB_NOTDIRTY;

        }

    }

}
