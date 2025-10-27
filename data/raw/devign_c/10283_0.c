void qxl_log_command(PCIQXLDevice *qxl, const char *ring, QXLCommandExt *ext)

{

    bool compat = ext->flags & QXL_COMMAND_FLAG_COMPAT;

    void *data;



    if (!qxl->cmdlog) {

        return;

    }

    fprintf(stderr, "%" PRId64 " qxl-%d/%s:", qemu_get_clock_ns(vm_clock),

            qxl->id, ring);

    fprintf(stderr, " cmd @ 0x%" PRIx64 " %s%s", ext->cmd.data,

            qxl_name(qxl_type, ext->cmd.type),

            compat ? "(compat)" : "");



    data = qxl_phys2virt(qxl, ext->cmd.data, ext->group_id);

    switch (ext->cmd.type) {

    case QXL_CMD_DRAW:

        if (!compat) {

            qxl_log_cmd_draw(qxl, data, ext->group_id);

        } else {

            qxl_log_cmd_draw_compat(qxl, data, ext->group_id);

        }

        break;

    case QXL_CMD_SURFACE:

        qxl_log_cmd_surface(qxl, data);

        break;

    case QXL_CMD_CURSOR:

        qxl_log_cmd_cursor(qxl, data, ext->group_id);

        break;

    }

    fprintf(stderr, "\n");

}
