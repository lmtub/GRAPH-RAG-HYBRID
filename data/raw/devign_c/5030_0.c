int qemu_loadvm_state(QEMUFile *f)

{

    LIST_HEAD(, LoadStateEntry) loadvm_handlers =

        LIST_HEAD_INITIALIZER(loadvm_handlers);

    LoadStateEntry *le, *new_le;

    uint8_t section_type;

    unsigned int v;

    int ret;



    v = qemu_get_be32(f);

    if (v != QEMU_VM_FILE_MAGIC)

        return -EINVAL;



    v = qemu_get_be32(f);

    if (v == QEMU_VM_FILE_VERSION_COMPAT) {

        fprintf(stderr, "SaveVM v2 format is obsolete and don't work anymore\n");

        return -ENOTSUP;

    }

    if (v != QEMU_VM_FILE_VERSION)

        return -ENOTSUP;



    while ((section_type = qemu_get_byte(f)) != QEMU_VM_EOF) {

        uint32_t instance_id, version_id, section_id;

        SaveStateEntry *se;

        char idstr[257];

        int len;



        switch (section_type) {

        case QEMU_VM_SECTION_START:

        case QEMU_VM_SECTION_FULL:

            /* Read section start */

            section_id = qemu_get_be32(f);

            len = qemu_get_byte(f);

            qemu_get_buffer(f, (uint8_t *)idstr, len);

            idstr[len] = 0;

            instance_id = qemu_get_be32(f);

            version_id = qemu_get_be32(f);



            /* Find savevm section */

            se = find_se(idstr, instance_id);

            if (se == NULL) {

                fprintf(stderr, "Unknown savevm section or instance '%s' %d\n", idstr, instance_id);

                ret = -EINVAL;

                goto out;

            }



            /* Validate version */

            if (version_id > se->version_id) {

                fprintf(stderr, "savevm: unsupported version %d for '%s' v%d\n",

                        version_id, idstr, se->version_id);

                ret = -EINVAL;

                goto out;

            }



            /* Add entry */

            le = qemu_mallocz(sizeof(*le));



            le->se = se;

            le->section_id = section_id;

            le->version_id = version_id;

            LIST_INSERT_HEAD(&loadvm_handlers, le, entry);



            ret = vmstate_load(f, le->se, le->version_id);

            if (ret < 0) {

                fprintf(stderr, "qemu: warning: error while loading state for instance 0x%x of device '%s'\n",

                        instance_id, idstr);

                goto out;

            }

            break;

        case QEMU_VM_SECTION_PART:

        case QEMU_VM_SECTION_END:

            section_id = qemu_get_be32(f);



            LIST_FOREACH(le, &loadvm_handlers, entry) {

                if (le->section_id == section_id) {

                    break;

                }

            }

            if (le == NULL) {

                fprintf(stderr, "Unknown savevm section %d\n", section_id);

                ret = -EINVAL;

                goto out;

            }



            ret = vmstate_load(f, le->se, le->version_id);

            if (ret < 0) {

                fprintf(stderr, "qemu: warning: error while loading state section id %d\n",

                        section_id);

                goto out;

            }

            break;

        default:

            fprintf(stderr, "Unknown savevm section type %d\n", section_type);

            ret = -EINVAL;

            goto out;

        }

    }



    ret = 0;



out:

    LIST_FOREACH_SAFE(le, &loadvm_handlers, entry, new_le) {

        LIST_REMOVE(le, entry);

        qemu_free(le);

    }



    if (qemu_file_has_error(f))

        ret = -EIO;



    return ret;

}
