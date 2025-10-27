static inline void tlb_protect_code1(CPUTLBEntry *tlb_entry, uint32_t addr)

{

    if (addr == (tlb_entry->address & 

                 (TARGET_PAGE_MASK | TLB_INVALID_MASK)) &&

        (tlb_entry->address & ~TARGET_PAGE_MASK) != IO_MEM_CODE) {

        tlb_entry->address |= IO_MEM_CODE;

        tlb_entry->addend -= (unsigned long)phys_ram_base;

    }

}
