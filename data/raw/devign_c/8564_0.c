void HELPER(dc_zva)(CPUARMState *env, uint64_t vaddr_in)

{

    /* Implement DC ZVA, which zeroes a fixed-length block of memory.

     * Note that we do not implement the (architecturally mandated)

     * alignment fault for attempts to use this on Device memory

     * (which matches the usual QEMU behaviour of not implementing either

     * alignment faults or any memory attribute handling).

     */



    ARMCPU *cpu = arm_env_get_cpu(env);

    uint64_t blocklen = 4 << cpu->dcz_blocksize;

    uint64_t vaddr = vaddr_in & ~(blocklen - 1);



#ifndef CONFIG_USER_ONLY

    {

        /* Slightly awkwardly, QEMU's TARGET_PAGE_SIZE may be less than

         * the block size so we might have to do more than one TLB lookup.

         * We know that in fact for any v8 CPU the page size is at least 4K

         * and the block size must be 2K or less, but TARGET_PAGE_SIZE is only

         * 1K as an artefact of legacy v5 subpage support being present in the

         * same QEMU executable.

         */

        int maxidx = DIV_ROUND_UP(blocklen, TARGET_PAGE_SIZE);

        void *hostaddr[maxidx];

        int try, i;

        unsigned mmu_idx = cpu_mmu_index(env, false);

        TCGMemOpIdx oi = make_memop_idx(MO_UB, mmu_idx);



        for (try = 0; try < 2; try++) {



            for (i = 0; i < maxidx; i++) {

                hostaddr[i] = tlb_vaddr_to_host(env,

                                                vaddr + TARGET_PAGE_SIZE * i,

                                                1, mmu_idx);

                if (!hostaddr[i]) {

                    break;

                }

            }

            if (i == maxidx) {

                /* If it's all in the TLB it's fair game for just writing to;

                 * we know we don't need to update dirty status, etc.

                 */

                for (i = 0; i < maxidx - 1; i++) {

                    memset(hostaddr[i], 0, TARGET_PAGE_SIZE);

                }

                memset(hostaddr[i], 0, blocklen - (i * TARGET_PAGE_SIZE));

                return;

            }

            /* OK, try a store and see if we can populate the tlb. This

             * might cause an exception if the memory isn't writable,

             * in which case we will longjmp out of here. We must for

             * this purpose use the actual register value passed to us

             * so that we get the fault address right.

             */

            helper_ret_stb_mmu(env, vaddr_in, 0, oi, GETRA());

            /* Now we can populate the other TLB entries, if any */

            for (i = 0; i < maxidx; i++) {

                uint64_t va = vaddr + TARGET_PAGE_SIZE * i;

                if (va != (vaddr_in & TARGET_PAGE_MASK)) {

                    helper_ret_stb_mmu(env, va, 0, oi, GETRA());

                }

            }

        }



        /* Slow path (probably attempt to do this to an I/O device or

         * similar, or clearing of a block of code we have translations

         * cached for). Just do a series of byte writes as the architecture

         * demands. It's not worth trying to use a cpu_physical_memory_map(),

         * memset(), unmap() sequence here because:

         *  + we'd need to account for the blocksize being larger than a page

         *  + the direct-RAM access case is almost always going to be dealt

         *    with in the fastpath code above, so there's no speed benefit

         *  + we would have to deal with the map returning NULL because the

         *    bounce buffer was in use

         */

        for (i = 0; i < blocklen; i++) {

            helper_ret_stb_mmu(env, vaddr + i, 0, oi, GETRA());

        }

    }

#else

    memset(g2h(vaddr), 0, blocklen);

#endif

}
