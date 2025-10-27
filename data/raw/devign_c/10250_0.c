static int translate_pages(S390CPU *cpu, vaddr addr, int nr_pages,

                           target_ulong *pages, bool is_write)

{

    bool lowprot = is_write && lowprot_enabled(&cpu->env);

    uint64_t asc = cpu->env.psw.mask & PSW_MASK_ASC;

    CPUS390XState *env = &cpu->env;

    int ret, i, pflags;



    for (i = 0; i < nr_pages; i++) {

        /* Low-address protection? */

        if (lowprot && (addr < 512 || (addr >= 4096 && addr < 4096 + 512))) {

            trigger_access_exception(env, PGM_PROTECTION, ILEN_AUTO, 0);

            return -EACCES;

        }

        ret = mmu_translate(env, addr, is_write, asc, &pages[i], &pflags, true);

        if (ret) {

            return ret;

        }

        if (!address_space_access_valid(&address_space_memory, pages[i],

                                        TARGET_PAGE_SIZE, is_write)) {

            program_interrupt(env, PGM_ADDRESSING, ILEN_AUTO);

            return -EFAULT;

        }

        addr += TARGET_PAGE_SIZE;

    }



    return 0;

}
