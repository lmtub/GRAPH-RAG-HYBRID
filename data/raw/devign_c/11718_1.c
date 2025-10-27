int qemu_savevm_state_complete(Monitor *mon, QEMUFile *f)
{
    SaveStateEntry *se;
    QTAILQ_FOREACH(se, &savevm_handlers, entry) {
        if (se->save_live_state == NULL)
            continue;
        /* Section type */
        qemu_put_byte(f, QEMU_VM_SECTION_END);
        qemu_put_be32(f, se->section_id);
        se->save_live_state(mon, f, QEMU_VM_SECTION_END, se->opaque);
    }
    QTAILQ_FOREACH(se, &savevm_handlers, entry) {
        int len;
	if (se->save_state == NULL && se->vmsd == NULL)
	    continue;
        /* Section type */
        qemu_put_byte(f, QEMU_VM_SECTION_FULL);
        qemu_put_be32(f, se->section_id);
        /* ID string */
        len = strlen(se->idstr);
        qemu_put_byte(f, len);
        qemu_put_buffer(f, (uint8_t *)se->idstr, len);
        qemu_put_be32(f, se->instance_id);
        qemu_put_be32(f, se->version_id);
        vmstate_save(f, se);
    }
    qemu_put_byte(f, QEMU_VM_EOF);
    if (qemu_file_has_error(f))
        return -EIO;
    return 0;
}