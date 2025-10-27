static void framebuffer_update_request(VncState *vs, int incremental,

				       int x_position, int y_position,

				       int w, int h)

{

    if (x_position > ds_get_width(vs->ds))

        x_position = ds_get_width(vs->ds);

    if (y_position > ds_get_height(vs->ds))

        y_position = ds_get_height(vs->ds);

    if (x_position + w >= ds_get_width(vs->ds))

        w = ds_get_width(vs->ds)  - x_position;

    if (y_position + h >= ds_get_height(vs->ds))

        h = ds_get_height(vs->ds) - y_position;



    int i;

    vs->need_update = 1;

    if (!incremental) {

	char *old_row = vs->old_data + y_position * ds_get_linesize(vs->ds);



	for (i = 0; i < h; i++) {

            vnc_set_bits(vs->dirty_row[y_position + i],

                         (ds_get_width(vs->ds) / 16), VNC_DIRTY_WORDS);

	    memset(old_row, 42, ds_get_width(vs->ds) * vs->depth);

	    old_row += ds_get_linesize(vs->ds);

	}

    }

}
