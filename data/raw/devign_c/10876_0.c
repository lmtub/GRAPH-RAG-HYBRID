hwaddr memory_region_section_get_iotlb(CPUArchState *env,

                                       MemoryRegionSection *section,

                                       target_ulong vaddr,

                                       hwaddr paddr, hwaddr xlat,

                                       int prot,

                                       target_ulong *address)

{

    hwaddr iotlb;

    CPUWatchpoint *wp;



    if (memory_region_is_ram(section->mr)) {

        /* Normal RAM.  */

        iotlb = (memory_region_get_ram_addr(section->mr) & TARGET_PAGE_MASK)

            + xlat;

        if (!section->readonly) {

            iotlb |= PHYS_SECTION_NOTDIRTY;

        } else {

            iotlb |= PHYS_SECTION_ROM;

        }

    } else {

        iotlb = section - address_space_memory.dispatch->sections;

        iotlb += xlat;

    }



    /* Make accesses to pages with watchpoints go via the

       watchpoint trap routines.  */

    QTAILQ_FOREACH(wp, &env->watchpoints, entry) {

        if (vaddr == (wp->vaddr & TARGET_PAGE_MASK)) {

            /* Avoid trapping reads of pages with a write breakpoint. */

            if ((prot & PAGE_WRITE) || (wp->flags & BP_MEM_READ)) {

                iotlb = PHYS_SECTION_WATCH + paddr;

                *address |= TLB_MMIO;

                break;

            }

        }

    }



    return iotlb;

}
