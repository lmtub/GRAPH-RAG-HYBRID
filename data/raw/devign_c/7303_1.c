static int qemu_savevm_state(QEMUFile *f)

{

    SaveStateEntry *se;

    int len, ret;

    int64_t cur_pos, len_pos, total_len_pos;



    qemu_put_be32(f, QEMU_VM_FILE_MAGIC);

    qemu_put_be32(f, QEMU_VM_FILE_VERSION);

    total_len_pos = qemu_ftell(f);

    qemu_put_be64(f, 0); /* total size */



    for(se = first_se; se != NULL; se = se->next) {

	if (se->save_state == NULL)

	    /* this one has a loader only, for backwards compatibility */

	    continue;



        /* ID string */

        len = strlen(se->idstr);

        qemu_put_byte(f, len);

        qemu_put_buffer(f, (uint8_t *)se->idstr, len);



        qemu_put_be32(f, se->instance_id);

        qemu_put_be32(f, se->version_id);



        /* record size: filled later */

        len_pos = qemu_ftell(f);

        qemu_put_be32(f, 0);

        se->save_state(f, se->opaque);



        /* fill record size */

        cur_pos = qemu_ftell(f);

        len = cur_pos - len_pos - 4;

        qemu_fseek(f, len_pos, SEEK_SET);

        qemu_put_be32(f, len);

        qemu_fseek(f, cur_pos, SEEK_SET);

    }

    cur_pos = qemu_ftell(f);

    qemu_fseek(f, total_len_pos, SEEK_SET);

    qemu_put_be64(f, cur_pos - total_len_pos - 8);

    qemu_fseek(f, cur_pos, SEEK_SET);



    ret = 0;

    return ret;

}
