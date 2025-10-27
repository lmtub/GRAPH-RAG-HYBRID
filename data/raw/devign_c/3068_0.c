static int dump_init(DumpState *s, int fd, bool has_format,

                     DumpGuestMemoryFormat format, bool paging, bool has_filter,

                     int64_t begin, int64_t length, Error **errp)

{

    CPUState *cpu;

    int nr_cpus;

    Error *err = NULL;

    int ret;



    /* kdump-compressed is conflict with paging and filter */

    if (has_format && format != DUMP_GUEST_MEMORY_FORMAT_ELF) {

        assert(!paging && !has_filter);

    }



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

    CPU_FOREACH(cpu) {

        nr_cpus++;

    }



    s->fd = fd;

    s->has_filter = has_filter;

    s->begin = begin;

    s->length = length;



    guest_phys_blocks_init(&s->guest_phys_blocks);

    guest_phys_blocks_append(&s->guest_phys_blocks);



    s->start = get_start_block(s);

    if (s->start == -1) {

        error_set(errp, QERR_INVALID_PARAMETER, "begin");

        goto cleanup;

    }



    /* get dump info: endian, class and architecture.

     * If the target architecture is not supported, cpu_get_dump_info() will

     * return -1.

     */

    ret = cpu_get_dump_info(&s->dump_info, &s->guest_phys_blocks);

    if (ret < 0) {

        error_set(errp, QERR_UNSUPPORTED);

        goto cleanup;

    }



    s->note_size = cpu_get_note_size(s->dump_info.d_class,

                                     s->dump_info.d_machine, nr_cpus);

    if (s->note_size < 0) {

        error_set(errp, QERR_UNSUPPORTED);

        goto cleanup;

    }



    /* get memory mapping */

    memory_mapping_list_init(&s->list);

    if (paging) {

        qemu_get_guest_memory_mapping(&s->list, &s->guest_phys_blocks, &err);

        if (err != NULL) {

            error_propagate(errp, err);

            goto cleanup;

        }

    } else {

        qemu_get_guest_simple_memory_mapping(&s->list, &s->guest_phys_blocks);

    }



    s->nr_cpus = nr_cpus;

    s->page_size = TARGET_PAGE_SIZE;



    get_max_mapnr(s);



    uint64_t tmp;

    tmp = DIV_ROUND_UP(DIV_ROUND_UP(s->max_mapnr, CHAR_BIT), s->page_size);

    s->len_dump_bitmap = tmp * s->page_size;



    /* init for kdump-compressed format */

    if (has_format && format != DUMP_GUEST_MEMORY_FORMAT_ELF) {

        switch (format) {

        case DUMP_GUEST_MEMORY_FORMAT_KDUMP_ZLIB:

            s->flag_compress = DUMP_DH_COMPRESSED_ZLIB;

            break;



        case DUMP_GUEST_MEMORY_FORMAT_KDUMP_LZO:

            s->flag_compress = DUMP_DH_COMPRESSED_LZO;

            break;



        case DUMP_GUEST_MEMORY_FORMAT_KDUMP_SNAPPY:

            s->flag_compress = DUMP_DH_COMPRESSED_SNAPPY;

            break;



        default:

            s->flag_compress = 0;

        }



        return 0;

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
