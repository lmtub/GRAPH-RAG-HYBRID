static inline void *alloc_code_gen_buffer(void)

{

    int flags = MAP_PRIVATE | MAP_ANONYMOUS;

    uintptr_t start = 0;

    size_t size = tcg_ctx.code_gen_buffer_size;

    void *buf;



    /* Constrain the position of the buffer based on the host cpu.

       Note that these addresses are chosen in concert with the

       addresses assigned in the relevant linker script file.  */

# if defined(__PIE__) || defined(__PIC__)

    /* Don't bother setting a preferred location if we're building

       a position-independent executable.  We're more likely to get

       an address near the main executable if we let the kernel

       choose the address.  */

# elif defined(__x86_64__) && defined(MAP_32BIT)

    /* Force the memory down into low memory with the executable.

       Leave the choice of exact location with the kernel.  */

    flags |= MAP_32BIT;

    /* Cannot expect to map more than 800MB in low memory.  */

    if (size > 800u * 1024 * 1024) {

        tcg_ctx.code_gen_buffer_size = size = 800u * 1024 * 1024;

    }

# elif defined(__sparc__)

    start = 0x40000000ul;

# elif defined(__s390x__)

    start = 0x90000000ul;

# elif defined(__mips__)

#  if _MIPS_SIM == _ABI64

    start = 0x128000000ul;

#  else

    start = 0x08000000ul;

#  endif

# endif



    buf = mmap((void *)start, size + qemu_real_host_page_size,

               PROT_NONE, flags, -1, 0);

    if (buf == MAP_FAILED) {

        return NULL;

    }



#ifdef __mips__

    if (cross_256mb(buf, size)) {

        /* Try again, with the original still mapped, to avoid re-acquiring

           that 256mb crossing.  This time don't specify an address.  */

        size_t size2;

        void *buf2 = mmap(NULL, size + qemu_real_host_page_size,

                          PROT_NONE, flags, -1, 0);

        switch (buf2 != MAP_FAILED) {

        case 1:

            if (!cross_256mb(buf2, size)) {

                /* Success!  Use the new buffer.  */

                munmap(buf, size);

                break;

            }

            /* Failure.  Work with what we had.  */

            munmap(buf2, size);

            /* fallthru */

        default:

            /* Split the original buffer.  Free the smaller half.  */

            buf2 = split_cross_256mb(buf, size);

            size2 = tcg_ctx.code_gen_buffer_size;

            if (buf == buf2) {

                munmap(buf + size2 + qemu_real_host_page_size, size - size2);

            } else {

                munmap(buf, size - size2);

            }

            size = size2;

            break;

        }

        buf = buf2;

    }

#endif



    /* Make the final buffer accessible.  The guard page at the end

       will remain inaccessible with PROT_NONE.  */

    mprotect(buf, size, PROT_WRITE | PROT_READ | PROT_EXEC);



    /* Request large pages for the buffer.  */

    qemu_madvise(buf, size, QEMU_MADV_HUGEPAGE);



    return buf;

}
