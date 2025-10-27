static int validate_guest_space(unsigned long guest_base,

                                unsigned long guest_size)

{

    unsigned long real_start, test_page_addr;



    /* We need to check that we can force a fault on access to the

     * commpage at 0xffff0fxx

     */

    test_page_addr = guest_base + (0xffff0f00 & qemu_host_page_mask);



    /* If the commpage lies within the already allocated guest space,

     * then there is no way we can allocate it.

     */

    if (test_page_addr >= guest_base

        && test_page_addr <= (guest_base + guest_size)) {

        return -1;

    }



    /* Note it needs to be writeable to let us initialise it */

    real_start = (unsigned long)

                 mmap((void *)test_page_addr, qemu_host_page_size,

                     PROT_READ | PROT_WRITE,

                     MAP_ANONYMOUS | MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);



    /* If we can't map it then try another address */

    if (real_start == -1ul) {

        return 0;

    }



    if (real_start != test_page_addr) {

        /* OS didn't put the page where we asked - unmap and reject */

        munmap((void *)real_start, qemu_host_page_size);

        return 0;

    }



    /* Leave the page mapped

     * Populate it (mmap should have left it all 0'd)

     */



    /* Kernel helper versions */

    __put_user(5, (uint32_t *)g2h(0xffff0ffcul));



    /* Now it's populated make it RO */

    if (mprotect((void *)test_page_addr, qemu_host_page_size, PROT_READ)) {

        perror("Protecting guest commpage");

        exit(-1);

    }



    return 1; /* All good */

}
