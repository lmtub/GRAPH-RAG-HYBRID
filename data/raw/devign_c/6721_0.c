static void release_unused_pictures(H264Context *h, int remove_current)

{

    int i;



    /* release non reference frames */

    for (i = 0; i < MAX_PICTURE_COUNT; i++) {

        if (h->DPB[i].f.data[0] && !h->DPB[i].reference &&

            (remove_current || &h->DPB[i] != h->cur_pic_ptr)) {

            unref_picture(h, &h->DPB[i]);

        }

    }

}
