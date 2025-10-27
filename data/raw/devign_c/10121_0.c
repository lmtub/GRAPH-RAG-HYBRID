static int dump_init(DumpState *s, int fd, bool paging, bool has_filter,

                     int64_t begin, int64_t length, Error **errp)

{

    CPUState *cpu;

    int nr_cpus;

    Error *err = NULL;

    int ret;



    if (runstate_is_running()) {

        vm_stop(RUN_STATE_SAVE_VM);

        s->resume = true;

    } else {

        s->resume = false;

    }



    /* If we use KVM, we should synchronize the registers before we get dump

     * info or physmap info.

     */

    cpu_synchronize_all_states();

    nr_cpus = 0;

    for (cpu = first_cpu; cpu != NULL; cpu = cpu->next_cpu) {

        nr_cpus++;

    }



    s->errp = errp;

    s->fd = fd;

    s->has_filter = has_filter;

    s->begin = begin;

    s->length = length;



    guest_phys_blocks_init(&s->guest_phys_blocks);

    /* FILL LIST */



    s->start = get_start_block(s);

    if (s->start == -1) {

        error_set(errp, QERR_INVALID_PARAMETER, "begin");

        goto cleanup;

    }



    /* get dump info: endian, class and architecture.

     * If the target architecture is not supported, cpu_get_dump_info() will

     * return -1.

     */

    ret = cpu_get_dump_info(&s->dump_info);

    if (ret < 0) {

        error_set(errp, QERR_UNSUPPORTED);

        goto cleanup;

    }



    s->note_size = cpu_get_note_size(s->dump_info.d_class,

                                     s->dump_info.d_machine, nr_cpus);

    if (ret < 0) {

        error_set(errp, QERR_UNSUPPORTED);

        goto cleanup;

    }



    /* get memory mapping */

    memory_mapping_list_init(&s->list);

    if (paging) {

        qemu_get_guest_memory_mapping(&s->list, &err);

        if (err != NULL) {

            error_propagate(errp, err);

            goto cleanup;

        }

    } else {

        qemu_get_guest_simple_memory_mapping(&s->list);

    }



    if (s->has_filter) {

        memory_mapping_filter(&s->list, s->begin, s->length);

    }



    /*

     * calculate phdr_num

     *

     * the type of ehdr->e_phnum is uint16_t, so we should avoid overflow

     */

    s->phdr_num = 1; /* PT_NOTE */

    if (s->list.num < UINT16_MAX - 2) {

        s->phdr_num += s->list.num;

        s->have_section = false;

    } else {

        s->have_section = true;

        s->phdr_num = PN_XNUM;

        s->sh_info = 1; /* PT_NOTE */



        /* the type of shdr->sh_info is uint32_t, so we should avoid overflow */

        if (s->list.num <= UINT32_MAX - 1) {

            s->sh_info += s->list.num;

        } else {

            s->sh_info = UINT32_MAX;

        }

    }



    if (s->dump_info.d_class == ELFCLASS64) {

        if (s->have_section) {

            s->memory_offset = sizeof(Elf64_Ehdr) +

                               sizeof(Elf64_Phdr) * s->sh_info +

                               sizeof(Elf64_Shdr) + s->note_size;

        } else {

            s->memory_offset = sizeof(Elf64_Ehdr) +

                               sizeof(Elf64_Phdr) * s->phdr_num + s->note_size;

        }

    } else {

        if (s->have_section) {

            s->memory_offset = sizeof(Elf32_Ehdr) +

                               sizeof(Elf32_Phdr) * s->sh_info +

                               sizeof(Elf32_Shdr) + s->note_size;

        } else {

            s->memory_offset = sizeof(Elf32_Ehdr) +

                               sizeof(Elf32_Phdr) * s->phdr_num + s->note_size;

        }

    }



    return 0;



cleanup:

    guest_phys_blocks_free(&s->guest_phys_blocks);



    if (s->resume) {

        vm_start();

    }



    return -1;

}
