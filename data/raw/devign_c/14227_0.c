void qxl_log_cmd_cursor(PCIQXLDevice *qxl, QXLCursorCmd *cmd, int group_id)

{

    QXLCursor *cursor;



    fprintf(stderr, ": %s",

            qxl_name(qxl_cursor_cmd, cmd->type));

    switch (cmd->type) {

    case QXL_CURSOR_SET:

        fprintf(stderr, " +%d+%d visible %s, shape @ 0x%" PRIx64,

                cmd->u.set.position.x,

                cmd->u.set.position.y,

                cmd->u.set.visible ? "yes" : "no",

                cmd->u.set.shape);

        cursor = qxl_phys2virt(qxl, cmd->u.set.shape, group_id);

        fprintf(stderr, " type %s size %dx%d hot-spot +%d+%d"

                " unique 0x%" PRIx64 " data-size %d",

                qxl_name(spice_cursor_type, cursor->header.type),

                cursor->header.width, cursor->header.height,

                cursor->header.hot_spot_x, cursor->header.hot_spot_y,

                cursor->header.unique, cursor->data_size);

        break;

    case QXL_CURSOR_MOVE:

        fprintf(stderr, " +%d+%d", cmd->u.position.x, cmd->u.position.y);

        break;

    }

}
