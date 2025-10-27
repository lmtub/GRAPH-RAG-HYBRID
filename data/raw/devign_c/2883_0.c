int smbios_entry_add(const char *t)

{

    char buf[1024];



    if (get_param_value(buf, sizeof(buf), "file", t)) {

        struct smbios_structure_header *header;

        struct smbios_table *table;

        int size = get_image_size(buf);



        if (size < sizeof(struct smbios_structure_header)) {

            fprintf(stderr, "Cannot read smbios file %s", buf);

            exit(1);

        }



        if (!smbios_entries) {

            smbios_entries_len = sizeof(uint16_t);

            smbios_entries = qemu_mallocz(smbios_entries_len);

        }



        smbios_entries = qemu_realloc(smbios_entries, smbios_entries_len +

                                                      sizeof(*table) + size);

        table = (struct smbios_table *)(smbios_entries + smbios_entries_len);

        table->header.type = SMBIOS_TABLE_ENTRY;

        table->header.length = cpu_to_le16(sizeof(*table) + size);



        if (load_image(buf, table->data) != size) {

            fprintf(stderr, "Failed to load smbios file %s", buf);

            exit(1);

        }



        header = (struct smbios_structure_header *)(table->data);

        smbios_check_collision(header->type, SMBIOS_TABLE_ENTRY);



        smbios_entries_len += sizeof(*table) + size;

        (*(uint16_t *)smbios_entries) =

                cpu_to_le16(le16_to_cpu(*(uint16_t *)smbios_entries) + 1);

        return 0;

    }



    if (get_param_value(buf, sizeof(buf), "type", t)) {

        unsigned long type = strtoul(buf, NULL, 0);

        switch (type) {

        case 0:

            smbios_build_type_0_fields(t);

            return 0;

        case 1:

            smbios_build_type_1_fields(t);

            return 0;

        default:

            fprintf(stderr, "Don't know how to build fields for SMBIOS type "

                    "%ld\n", type);

            exit(1);

        }

    }



    fprintf(stderr, "smbios: must specify type= or file=\n");

    return -1;

}
