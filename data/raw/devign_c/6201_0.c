target_phys_addr_t memory_region_section_get_iotlb(CPUArchState *env,

                                                   MemoryRegionSection *section,

                                                   target_ulong vaddr,

                                                   target_phys_addr_t paddr,

                                                   int prot,

                                                   target_ulong *address)

{

    target_phys_addr_t iotlb;

    CPUWatchpoint *wp;



    if (memory_region_is_ram(section->mr)) {

        /* Normal RAM.  */

        iotlb = (memory_region_get_ram_addr(section->mr) & TARGET_PAGE_MASK)

            + memory_region_section_addr(section, paddr);

        if (!section->readonly) {

            iotlb |= phys_section_notdirty;

        } else {

            iotlb |= phys_section_rom;

        }

    } else {

        /* IO handlers are currently passed a physical address.

           It would be nice to pass an offset from the base address

           of that region.  This would avoid having to special case RAM,

           and avoid full address decoding in every device.

           We can't use the high bits of pd for this because

           IO_MEM_ROMD uses these as a ram address.  */

        iotlb = section - phys_sections;

        iotlb += memory_region_section_addr(section, paddr);

    }



    /* Make accesses to pages with watchpoints go via the

       watchpoint trap routines.  */

    QTAILQ_FOREACH(wp, &env->watchpoints, entry) {

        if (vaddr == (wp->vaddr & TARGET_PAGE_MASK)) {

            /* Avoid trapping reads of pages with a write breakpoint. */

            if ((prot & PAGE_WRITE) || (wp->flags & BP_MEM_READ)) {

                iotlb = phys_section_watch + paddr;

                *address |= TLB_MMIO;

                break;

            }

        }

    }



    return iotlb;

}
