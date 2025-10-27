static bool tlb_is_dirty_ram(CPUTLBEntry *tlbe)

{

    return (tlbe->addr_write & (TLB_INVALID_MASK|TLB_MMIO|TLB_NOTDIRTY)) == 0;

}
