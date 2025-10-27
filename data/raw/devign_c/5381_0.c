void tlb_set_page(CPUState *env, target_ulong vaddr,

                  target_phys_addr_t paddr, int prot,

                  int mmu_idx, target_ulong size)

{

    PhysPageDesc *p;

    unsigned long pd;

    unsigned int index;

    target_ulong address;

    target_ulong code_address;

    unsigned long addend;

    CPUTLBEntry *te;

    CPUWatchpoint *wp;

    target_phys_addr_t iotlb;



    assert(size >= TARGET_PAGE_SIZE);

    if (size != TARGET_PAGE_SIZE) {

        tlb_add_large_page(env, vaddr, size);

    }

    p = phys_page_find(paddr >> TARGET_PAGE_BITS);

    if (!p) {

        pd = IO_MEM_UNASSIGNED;

    } else {

        pd = p->phys_offset;

    }

#if defined(DEBUG_TLB)

    printf("tlb_set_page: vaddr=" TARGET_FMT_lx " paddr=0x%08x prot=%x idx=%d smmu=%d pd=0x%08lx\n",

           vaddr, (int)paddr, prot, mmu_idx, is_softmmu, pd);

#endif



    address = vaddr;

    if ((pd & ~TARGET_PAGE_MASK) > IO_MEM_ROM && !(pd & IO_MEM_ROMD)) {

        /* IO memory case (romd handled later) */

        address |= TLB_MMIO;

    }

    addend = (unsigned long)qemu_get_ram_ptr(pd & TARGET_PAGE_MASK);

    if ((pd & ~TARGET_PAGE_MASK) <= IO_MEM_ROM) {

        /* Normal RAM.  */

        iotlb = pd & TARGET_PAGE_MASK;

        if ((pd & ~TARGET_PAGE_MASK) == IO_MEM_RAM)

            iotlb |= IO_MEM_NOTDIRTY;

        else

            iotlb |= IO_MEM_ROM;

    } else {

        /* IO handlers are currently passed a physical address.

           It would be nice to pass an offset from the base address

           of that region.  This would avoid having to special case RAM,

           and avoid full address decoding in every device.

           We can't use the high bits of pd for this because

           IO_MEM_ROMD uses these as a ram address.  */

        iotlb = (pd & ~TARGET_PAGE_MASK);

        if (p) {

            iotlb += p->region_offset;

        } else {

            iotlb += paddr;

        }

    }



    code_address = address;

    /* Make accesses to pages with watchpoints go via the

       watchpoint trap routines.  */

    QTAILQ_FOREACH(wp, &env->watchpoints, entry) {

        if (vaddr == (wp->vaddr & TARGET_PAGE_MASK)) {

            iotlb = io_mem_watch + paddr;

            /* TODO: The memory case can be optimized by not trapping

               reads of pages with a write breakpoint.  */

            address |= TLB_MMIO;

        }

    }



    index = (vaddr >> TARGET_PAGE_BITS) & (CPU_TLB_SIZE - 1);

    env->iotlb[mmu_idx][index] = iotlb - vaddr;

    te = &env->tlb_table[mmu_idx][index];

    te->addend = addend - vaddr;

    if (prot & PAGE_READ) {

        te->addr_read = address;

    } else {

        te->addr_read = -1;

    }



    if (prot & PAGE_EXEC) {

        te->addr_code = code_address;

    } else {

        te->addr_code = -1;

    }

    if (prot & PAGE_WRITE) {

        if ((pd & ~TARGET_PAGE_MASK) == IO_MEM_ROM ||

            (pd & IO_MEM_ROMD)) {

            /* Write access calls the I/O callback.  */

            te->addr_write = address | TLB_MMIO;

        } else if ((pd & ~TARGET_PAGE_MASK) == IO_MEM_RAM &&

                   !cpu_physical_memory_is_dirty(pd)) {

            te->addr_write = address | TLB_NOTDIRTY;

        } else {

            te->addr_write = address;

        }

    } else {

        te->addr_write = -1;

    }

}
