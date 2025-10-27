static int find_unused_picture(MpegEncContext *s, int shared)

{

    int i;



    if (shared) {

        for (i = 0; i < MAX_PICTURE_COUNT; i++) {

            if (s->picture[i].f.data[0] == NULL)

                return i;

        }

    } else {

        for (i = 0; i < MAX_PICTURE_COUNT; i++) {

            if (pic_is_unused(s, &s->picture[i]))

                return i;

        }

    }



    return AVERROR_INVALIDDATA;

}
