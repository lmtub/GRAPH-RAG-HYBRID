void smbios_entry_add(QemuOpts *opts)

{

    Error *local_err = NULL;

    const char *val;



    assert(!smbios_immutable);

    val = qemu_opt_get(opts, "file");

    if (val) {

        struct smbios_structure_header *header;

        struct smbios_table *table;

        int size;



        qemu_opts_validate(opts, qemu_smbios_file_opts, &local_err);

        if (local_err) {

            error_report("%s", error_get_pretty(local_err));

            exit(1);

        }



        size = get_image_size(val);

        if (size == -1 || size < sizeof(struct smbios_structure_header)) {

            error_report("Cannot read SMBIOS file %s", val);

            exit(1);

        }



        if (!smbios_entries) {

            smbios_entries_len = sizeof(uint16_t);

            smbios_entries = g_malloc0(smbios_entries_len);

        }



        smbios_entries = g_realloc(smbios_entries, smbios_entries_len +

                                                      sizeof(*table) + size);

        table = (struct smbios_table *)(smbios_entries + smbios_entries_len);

        table->header.type = SMBIOS_TABLE_ENTRY;

        table->header.length = cpu_to_le16(sizeof(*table) + size);



        if (load_image(val, table->data) != size) {

            error_report("Failed to load SMBIOS file %s", val);

            exit(1);

        }



        header = (struct smbios_structure_header *)(table->data);



        if (test_bit(header->type, have_fields_bitmap)) {

            error_report("can't load type %d struct, fields already specified!",

                         header->type);

            exit(1);

        }

        set_bit(header->type, have_binfile_bitmap);



        if (header->type == 4) {

            smbios_type4_count++;

        }



        smbios_entries_len += sizeof(*table) + size;

        (*(uint16_t *)smbios_entries) =

                cpu_to_le16(le16_to_cpu(*(uint16_t *)smbios_entries) + 1);

        return;

    }



    val = qemu_opt_get(opts, "type");

    if (val) {

        unsigned long type = strtoul(val, NULL, 0);



        if (type > SMBIOS_MAX_TYPE) {

            error_report("out of range!");

            exit(1);

        }



        if (test_bit(type, have_binfile_bitmap)) {

            error_report("can't add fields, binary file already loaded!");

            exit(1);

        }

        set_bit(type, have_fields_bitmap);



        switch (type) {

        case 0:

            qemu_opts_validate(opts, qemu_smbios_type0_opts, &local_err);

            if (local_err) {

                error_report("%s", error_get_pretty(local_err));

                exit(1);

            }

            save_opt(&type0.vendor, opts, "vendor");

            save_opt(&type0.version, opts, "version");

            save_opt(&type0.date, opts, "date");



            val = qemu_opt_get(opts, "release");

            if (val) {

                if (sscanf(val, "%hhu.%hhu", &type0.major, &type0.minor) != 2) {

                    error_report("Invalid release");

                    exit(1);

                }

                type0.have_major_minor = true;

            }

            return;

        case 1:

            qemu_opts_validate(opts, qemu_smbios_type1_opts, &local_err);

            if (local_err) {

                error_report("%s", error_get_pretty(local_err));

                exit(1);

            }

            save_opt(&type1.manufacturer, opts, "manufacturer");

            save_opt(&type1.product, opts, "product");

            save_opt(&type1.version, opts, "version");

            save_opt(&type1.serial, opts, "serial");

            save_opt(&type1.sku, opts, "sku");

            save_opt(&type1.family, opts, "family");



            val = qemu_opt_get(opts, "uuid");

            if (val) {

                if (qemu_uuid_parse(val, qemu_uuid) != 0) {

                    error_report("Invalid UUID");

                    exit(1);

                }

                qemu_uuid_set = true;

            }

            return;

        default:

            error_report("Don't know how to build fields for SMBIOS type %ld",

                         type);

            exit(1);

        }

    }



    error_report("Must specify type= or file=");

    exit(1);

}
