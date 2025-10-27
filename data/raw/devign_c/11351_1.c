static int find_image_range(int *pfirst_index, int *plast_index,

                            const char *path, int start_index)

{

    char buf[1024];

    int range, last_index, range1, first_index;



    /* find the first image */

    for (first_index = start_index; first_index < start_index + 5; first_index++) {

        if (av_get_frame_filename(buf, sizeof(buf), path, first_index) < 0){

            *pfirst_index =

            *plast_index = 1;

            if (avio_check(buf, AVIO_FLAG_READ) > 0)

                return 0;

            return -1;

        }

        if (avio_check(buf, AVIO_FLAG_READ) > 0)

            break;

    }

    if (first_index == 5)

        goto fail;



    /* find the last image */

    last_index = first_index;

    for(;;) {

        range = 0;

        for(;;) {

            if (!range)

                range1 = 1;

            else

                range1 = 2 * range;

            if (av_get_frame_filename(buf, sizeof(buf), path,

                                      last_index + range1) < 0)

                goto fail;

            if (avio_check(buf, AVIO_FLAG_READ) <= 0)

                break;

            range = range1;

            /* just in case... */

            if (range >= (1 << 30))

                goto fail;

        }

        /* we are sure than image last_index + range exists */

        if (!range)

            break;

        last_index += range;

    }

    *pfirst_index = first_index;

    *plast_index = last_index;

    return 0;

 fail:

    return -1;

}
