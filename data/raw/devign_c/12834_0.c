static void qxl_log_cmd_draw_compat(PCIQXLDevice *qxl, QXLCompatDrawable *draw,

                                    int group_id)

{

    fprintf(stderr, ": type %s effect %s",

            qxl_name(qxl_draw_type, draw->type),

            qxl_name(qxl_draw_effect, draw->effect));

    if (draw->bitmap_offset) {

        fprintf(stderr, ": bitmap %d",

                draw->bitmap_offset);

        qxl_log_rect(&draw->bitmap_area);

    }

    switch (draw->type) {

    case QXL_DRAW_COPY:

        qxl_log_cmd_draw_copy(qxl, &draw->u.copy, group_id);

        break;

    }

}
