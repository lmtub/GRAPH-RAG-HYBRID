void kvm_setup_guest_memory(void *start, size_t size)

{

#ifdef CONFIG_VALGRIND_H

    VALGRIND_MAKE_MEM_DEFINED(start, size);

#endif

    if (!kvm_has_sync_mmu()) {

        int ret = qemu_madvise(start, size, QEMU_MADV_DONTFORK);



        if (ret) {

            perror("qemu_madvise");

            fprintf(stderr,

                    "Need MADV_DONTFORK in absence of synchronous KVM MMU\n");

            exit(1);

        }

    }

}
