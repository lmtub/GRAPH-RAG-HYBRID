static void qxl_log_cmd_draw_copy(PCIQXLDevice *qxl, QXLCopy *copy, int group_id)

{

    fprintf(stderr, " src %" PRIx64,

            copy->src_bitmap);

    qxl_log_image(qxl, copy->src_bitmap, group_id);

    fprintf(stderr, " area");

    qxl_log_rect(&copy->src_area);

    fprintf(stderr, " rop %d", copy->rop_descriptor);

}
