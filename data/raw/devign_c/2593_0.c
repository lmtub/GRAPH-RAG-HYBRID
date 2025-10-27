int qemu_savevm_state_begin(QEMUFile *f)

{

    SaveStateEntry *se;



    qemu_put_be32(f, QEMU_VM_FILE_MAGIC);

    qemu_put_be32(f, QEMU_VM_FILE_VERSION);



    TAILQ_FOREACH(se, &savevm_handlers, entry) {

        int len;



        if (se->save_live_state == NULL)

            continue;



        /* Section type */

        qemu_put_byte(f, QEMU_VM_SECTION_START);

        qemu_put_be32(f, se->section_id);



        /* ID string */

        len = strlen(se->idstr);

        qemu_put_byte(f, len);

        qemu_put_buffer(f, (uint8_t *)se->idstr, len);



        qemu_put_be32(f, se->instance_id);

        qemu_put_be32(f, se->version_id);



        se->save_live_state(f, QEMU_VM_SECTION_START, se->opaque);

    }



    if (qemu_file_has_error(f))

        return -EIO;



    return 0;

}
