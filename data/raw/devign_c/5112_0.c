static inline void tlb_update_dirty(CPUTLBEntry *tlb_entry)

{

    ram_addr_t ram_addr;

    void *p;



    if ((tlb_entry->addr_write & ~TARGET_PAGE_MASK) == io_mem_ram.ram_addr) {

        p = (void *)(unsigned long)((tlb_entry->addr_write & TARGET_PAGE_MASK)

            + tlb_entry->addend);

        ram_addr = qemu_ram_addr_from_host_nofail(p);

        if (!cpu_physical_memory_is_dirty(ram_addr)) {

            tlb_entry->addr_write |= TLB_NOTDIRTY;

        }

    }

}
