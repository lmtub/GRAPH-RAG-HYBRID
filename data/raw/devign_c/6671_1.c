int tlb_set_page_exec(CPUState *env, target_ulong vaddr, 

                      target_phys_addr_t paddr, int prot, 

                      int is_user, int is_softmmu)

{

    PhysPageDesc *p;

    unsigned long pd;

    unsigned int index;

    target_ulong address;

    target_phys_addr_t addend;

    int ret;

    CPUTLBEntry *te;




    p = phys_page_find(paddr >> TARGET_PAGE_BITS);

    if (!p) {

        pd = IO_MEM_UNASSIGNED;

    } else {

        pd = p->phys_offset;

    }

#if defined(DEBUG_TLB)

    printf("tlb_set_page: vaddr=" TARGET_FMT_lx " paddr=0x%08x prot=%x u=%d smmu=%d pd=0x%08lx\n",

           vaddr, (int)paddr, prot, is_user, is_softmmu, pd);

#endif



    ret = 0;

#if !defined(CONFIG_SOFTMMU)

    if (is_softmmu) 

#endif

    {

        if ((pd & ~TARGET_PAGE_MASK) > IO_MEM_ROM && !(pd & IO_MEM_ROMD)) {

            /* IO memory case */

            address = vaddr | pd;

            addend = paddr;

        } else {

            /* standard memory */

            address = vaddr;

            addend = (unsigned long)phys_ram_base + (pd & TARGET_PAGE_MASK);

        }



        /* Make accesses to pages with watchpoints go via the

           watchpoint trap routines.  */

        for (i = 0; i < env->nb_watchpoints; i++) {

            if (vaddr == (env->watchpoint[i].vaddr & TARGET_PAGE_MASK)) {

                if (address & ~TARGET_PAGE_MASK) {

                    env->watchpoint[i].is_ram = 0;

                    address = vaddr | io_mem_watch;

                } else {

                    env->watchpoint[i].is_ram = 1;

                    /* TODO: Figure out how to make read watchpoints coexist

                       with code.  */

                    pd = (pd & TARGET_PAGE_MASK) | io_mem_watch | IO_MEM_ROMD;

                }

            }

        }

        

        index = (vaddr >> TARGET_PAGE_BITS) & (CPU_TLB_SIZE - 1);

        addend -= vaddr;

        te = &env->tlb_table[is_user][index];

        te->addend = addend;

        if (prot & PAGE_READ) {

            te->addr_read = address;

        } else {

            te->addr_read = -1;

        }

        if (prot & PAGE_EXEC) {

            te->addr_code = address;

        } else {

            te->addr_code = -1;

        }

        if (prot & PAGE_WRITE) {

            if ((pd & ~TARGET_PAGE_MASK) == IO_MEM_ROM || 

                (pd & IO_MEM_ROMD)) {

                /* write access calls the I/O callback */

                te->addr_write = vaddr | 

                    (pd & ~(TARGET_PAGE_MASK | IO_MEM_ROMD));

            } else if ((pd & ~TARGET_PAGE_MASK) == IO_MEM_RAM && 

                       !cpu_physical_memory_is_dirty(pd)) {

                te->addr_write = vaddr | IO_MEM_NOTDIRTY;

            } else {

                te->addr_write = address;

            }

        } else {

            te->addr_write = -1;

        }

    }

#if !defined(CONFIG_SOFTMMU)

    else {

        if ((pd & ~TARGET_PAGE_MASK) > IO_MEM_ROM) {

            /* IO access: no mapping is done as it will be handled by the

               soft MMU */

            if (!(env->hflags & HF_SOFTMMU_MASK))

                ret = 2;

        } else {

            void *map_addr;



            if (vaddr >= MMAP_AREA_END) {

                ret = 2;

            } else {

                if (prot & PROT_WRITE) {

                    if ((pd & ~TARGET_PAGE_MASK) == IO_MEM_ROM || 

#if defined(TARGET_HAS_SMC) || 1

                        first_tb ||

#endif

                        ((pd & ~TARGET_PAGE_MASK) == IO_MEM_RAM && 

                         !cpu_physical_memory_is_dirty(pd))) {

                        /* ROM: we do as if code was inside */

                        /* if code is present, we only map as read only and save the

                           original mapping */

                        VirtPageDesc *vp;

                        

                        vp = virt_page_find_alloc(vaddr >> TARGET_PAGE_BITS, 1);

                        vp->phys_addr = pd;

                        vp->prot = prot;

                        vp->valid_tag = virt_valid_tag;

                        prot &= ~PAGE_WRITE;

                    }

                }

                map_addr = mmap((void *)vaddr, TARGET_PAGE_SIZE, prot, 

                                MAP_SHARED | MAP_FIXED, phys_ram_fd, (pd & TARGET_PAGE_MASK));

                if (map_addr == MAP_FAILED) {

                    cpu_abort(env, "mmap failed when mapped physical address 0x%08x to virtual address 0x%08x\n",

                              paddr, vaddr);

                }

            }

        }

    }

#endif

    return ret;

}