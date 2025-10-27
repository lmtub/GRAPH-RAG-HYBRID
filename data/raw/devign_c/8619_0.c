abi_long target_mremap(abi_ulong old_addr, abi_ulong old_size,

                       abi_ulong new_size, unsigned long flags,

                       abi_ulong new_addr)

{

    int prot;

    void *host_addr;



    mmap_lock();



    if (flags & MREMAP_FIXED) {

        host_addr = (void *) syscall(__NR_mremap, g2h(old_addr),

                                     old_size, new_size,

                                     flags,

                                     g2h(new_addr));



        if (RESERVED_VA && host_addr != MAP_FAILED) {

            /* If new and old addresses overlap then the above mremap will

               already have failed with EINVAL.  */

            mmap_reserve(old_addr, old_size);

        }

    } else if (flags & MREMAP_MAYMOVE) {

        abi_ulong mmap_start;



        mmap_start = mmap_find_vma(0, new_size);



        if (mmap_start == -1) {

            errno = ENOMEM;

            host_addr = MAP_FAILED;

        } else {

            host_addr = (void *) syscall(__NR_mremap, g2h(old_addr),

                                         old_size, new_size,

                                         flags | MREMAP_FIXED,

                                         g2h(mmap_start));

            mmap_reserve(old_addr, old_size);

        }

    } else {

        int prot = 0;

        if (RESERVED_VA && old_size < new_size) {

            abi_ulong addr;

            for (addr = old_addr + old_size;

                 addr < old_addr + new_size;

                 addr++) {

                prot |= page_get_flags(addr);

            }

        }

        if (prot == 0) {

            host_addr = mremap(g2h(old_addr), old_size, new_size, flags);

            if (host_addr != MAP_FAILED && RESERVED_VA && old_size > new_size) {

                mmap_reserve(old_addr + old_size, new_size - old_size);

            }

        } else {

            errno = ENOMEM;

            host_addr = MAP_FAILED;

        }

        /* Check if address fits target address space */

        if ((unsigned long)host_addr + new_size > (abi_ulong)-1) {

            /* Revert mremap() changes */

            host_addr = mremap(g2h(old_addr), new_size, old_size, flags);

            errno = ENOMEM;

            host_addr = MAP_FAILED;

        }

    }



    if (host_addr == MAP_FAILED) {

        new_addr = -1;

    } else {

        new_addr = h2g(host_addr);

        prot = page_get_flags(old_addr);

        page_set_flags(old_addr, old_addr + old_size, 0);

        page_set_flags(new_addr, new_addr + new_size, prot | PAGE_VALID);

    }

    mmap_unlock();

    return new_addr;

}
