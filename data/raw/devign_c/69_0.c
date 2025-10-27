int qxl_render_cursor(PCIQXLDevice *qxl, QXLCommandExt *ext)

{

    QXLCursorCmd *cmd = qxl_phys2virt(qxl, ext->cmd.data, ext->group_id);

    QXLCursor *cursor;

    QEMUCursor *c;



    if (!cmd) {

        return 1;

    }



    if (!dpy_cursor_define_supported(qxl->vga.con)) {

        return 0;

    }



    if (qxl->debug > 1 && cmd->type != QXL_CURSOR_MOVE) {

        fprintf(stderr, "%s", __FUNCTION__);

        qxl_log_cmd_cursor(qxl, cmd, ext->group_id);

        fprintf(stderr, "\n");

    }

    switch (cmd->type) {

    case QXL_CURSOR_SET:

        cursor = qxl_phys2virt(qxl, cmd->u.set.shape, ext->group_id);

        if (!cursor) {

            return 1;

        }

        c = qxl_cursor(qxl, cursor, ext->group_id);

        if (c == NULL) {

            c = cursor_builtin_left_ptr();

        }

        qemu_mutex_lock(&qxl->ssd.lock);

        if (qxl->ssd.cursor) {

            cursor_put(qxl->ssd.cursor);

        }

        qxl->ssd.cursor = c;

        qxl->ssd.mouse_x = cmd->u.set.position.x;

        qxl->ssd.mouse_y = cmd->u.set.position.y;

        qemu_mutex_unlock(&qxl->ssd.lock);

        qemu_bh_schedule(qxl->ssd.cursor_bh);

        break;

    case QXL_CURSOR_MOVE:

        qemu_mutex_lock(&qxl->ssd.lock);

        qxl->ssd.mouse_x = cmd->u.position.x;

        qxl->ssd.mouse_y = cmd->u.position.y;

        qemu_mutex_unlock(&qxl->ssd.lock);

        qemu_bh_schedule(qxl->ssd.cursor_bh);

        break;

    }

    return 0;

}
