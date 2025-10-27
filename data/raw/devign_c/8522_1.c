void fw_cfg_add_file_callback(FWCfgState *s,  const char *filename,

                              FWCfgCallback select_cb,

                              FWCfgWriteCallback write_cb,

                              void *callback_opaque,

                              void *data, size_t len, bool read_only)

{

    int i, index, count;

    size_t dsize;

    MachineClass *mc = MACHINE_GET_CLASS(qdev_get_machine());

    int order = 0;



    if (!s->files) {

        dsize = sizeof(uint32_t) + sizeof(FWCfgFile) * fw_cfg_file_slots(s);

        s->files = g_malloc0(dsize);

        fw_cfg_add_bytes(s, FW_CFG_FILE_DIR, s->files, dsize);

    }



    count = be32_to_cpu(s->files->count);

    assert(count < fw_cfg_file_slots(s));



    /* Find the insertion point. */

    if (mc->legacy_fw_cfg_order) {

        /*

         * Sort by order. For files with the same order, we keep them

         * in the sequence in which they were added.

         */

        order = get_fw_cfg_order(s, filename);

        for (index = count;

             index > 0 && order < s->entry_order[index - 1];

             index--);

    } else {

        /* Sort by file name. */

        for (index = count;

             index > 0 && strcmp(filename, s->files->f[index - 1].name) < 0;

             index--);

    }



    /*

     * Move all the entries from the index point and after down one

     * to create a slot for the new entry.  Because calculations are

     * being done with the index, make it so that "i" is the current

     * index and "i - 1" is the one being copied from, thus the

     * unusual start and end in the for statement.

     */

    for (i = count + 1; i > index; i--) {

        s->files->f[i] = s->files->f[i - 1];

        s->files->f[i].select = cpu_to_be16(FW_CFG_FILE_FIRST + i);

        s->entries[0][FW_CFG_FILE_FIRST + i] =

            s->entries[0][FW_CFG_FILE_FIRST + i - 1];

        s->entry_order[i] = s->entry_order[i - 1];

    }



    memset(&s->files->f[index], 0, sizeof(FWCfgFile));

    memset(&s->entries[0][FW_CFG_FILE_FIRST + index], 0, sizeof(FWCfgEntry));



    pstrcpy(s->files->f[index].name, sizeof(s->files->f[index].name), filename);

    for (i = 0; i <= count; i++) {

        if (i != index &&

            strcmp(s->files->f[index].name, s->files->f[i].name) == 0) {

            error_report("duplicate fw_cfg file name: %s",

                         s->files->f[index].name);

            exit(1);

        }

    }



    fw_cfg_add_bytes_callback(s, FW_CFG_FILE_FIRST + index,

                              select_cb, write_cb,

                              callback_opaque, data, len,

                              read_only);



    s->files->f[index].size   = cpu_to_be32(len);

    s->files->f[index].select = cpu_to_be16(FW_CFG_FILE_FIRST + index);

    s->entry_order[index] = order;

    trace_fw_cfg_add_file(s, index, s->files->f[index].name, len);



    s->files->count = cpu_to_be32(count+1);

}
