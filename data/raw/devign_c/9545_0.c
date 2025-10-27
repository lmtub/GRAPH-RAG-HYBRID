static void apply_loop_filter(Vp3DecodeContext *s)

{

    int x, y, plane;

    int width, height;

    int fragment;

    int stride;

    unsigned char *plane_data;



    int bounding_values_array[256];

    int *bounding_values= bounding_values_array+127;

    int filter_limit;



    /* find the right loop limit value */

    for (x = 63; x >= 0; x--) {

        if (vp31_ac_scale_factor[x] >= s->quality_index)

            break;

    }

    filter_limit = vp31_filter_limit_values[s->quality_index];



    /* set up the bounding values */

    memset(bounding_values_array, 0, 256 * sizeof(int));

    for (x = 0; x < filter_limit; x++) {

        bounding_values[-x - filter_limit] = -filter_limit + x;

        bounding_values[-x] = -x;

        bounding_values[x] = x;

        bounding_values[x + filter_limit] = filter_limit - x;

    }



    for (plane = 0; plane < 3; plane++) {



        if (plane == 0) {

            /* Y plane parameters */

            fragment = 0;

            width = s->fragment_width;

            height = s->fragment_height;

            stride = s->current_frame.linesize[0];

            plane_data = s->current_frame.data[0];

        } else if (plane == 1) {

            /* U plane parameters */

            fragment = s->u_fragment_start;

            width = s->fragment_width / 2;

            height = s->fragment_height / 2;

            stride = s->current_frame.linesize[1];

            plane_data = s->current_frame.data[1];

        } else {

            /* V plane parameters */

            fragment = s->v_fragment_start;

            width = s->fragment_width / 2;

            height = s->fragment_height / 2;

            stride = s->current_frame.linesize[2];

            plane_data = s->current_frame.data[2];

        }



        for (y = 0; y < height; y++) {



            for (x = 0; x < width; x++) {

START_TIMER

                /* do not perform left edge filter for left columns frags */

                if ((x > 0) &&

                    (s->all_fragments[fragment].coding_method != MODE_COPY)) {

                    horizontal_filter(

                        plane_data + s->all_fragments[fragment].first_pixel - 7*stride, 

                        stride, bounding_values);

                }



                /* do not perform top edge filter for top row fragments */

                if ((y > 0) &&

                    (s->all_fragments[fragment].coding_method != MODE_COPY)) {

                    vertical_filter(

                        plane_data + s->all_fragments[fragment].first_pixel + stride, 

                        stride, bounding_values);

                }



                /* do not perform right edge filter for right column

                 * fragments or if right fragment neighbor is also coded

                 * in this frame (it will be filtered in next iteration) */

                if ((x < width - 1) &&

                    (s->all_fragments[fragment].coding_method != MODE_COPY) &&

                    (s->all_fragments[fragment + 1].coding_method == MODE_COPY)) {

                    horizontal_filter(

                        plane_data + s->all_fragments[fragment + 1].first_pixel - 7*stride, 

                        stride, bounding_values);

                }



                /* do not perform bottom edge filter for bottom row

                 * fragments or if bottom fragment neighbor is also coded

                 * in this frame (it will be filtered in the next row) */

                if ((y < height - 1) &&

                    (s->all_fragments[fragment].coding_method != MODE_COPY) &&

                    (s->all_fragments[fragment + width].coding_method == MODE_COPY)) {

                    vertical_filter(

                        plane_data + s->all_fragments[fragment + width].first_pixel + stride, 

                        stride, bounding_values);

                }



                fragment++;

STOP_TIMER("loop filter")

            }

        }

    }

}
