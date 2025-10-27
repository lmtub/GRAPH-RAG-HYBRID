int qemu_savevm_state_iterate(QEMUFile *f)

{

    SaveStateEntry *se;

    int ret = 1;



    TAILQ_FOREACH(se, &savevm_handlers, entry) {

        if (se->save_live_state == NULL)

            continue;



        /* Section type */

        qemu_put_byte(f, QEMU_VM_SECTION_PART);

        qemu_put_be32(f, se->section_id);



        ret &= !!se->save_live_state(f, QEMU_VM_SECTION_PART, se->opaque);

    }



    if (ret)

        return 1;



    if (qemu_file_has_error(f))

        return -EIO;



    return 0;

}
