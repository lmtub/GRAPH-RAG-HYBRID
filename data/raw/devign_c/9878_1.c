static int read_uncompressed_sgi(const SGIInfo *si, 

        AVPicture *pict, ByteIOContext *f)

{

    int x, y, z, chan_offset, ret = 0;

    uint8_t *dest_row, *tmp_row = NULL;



    tmp_row = av_malloc(si->xsize);



    /* skip header */ 

    url_fseek(f, SGI_HEADER_SIZE, SEEK_SET);



    pict->linesize[0] = si->xsize;



    for (z = 0; z < si->zsize; z++) {



#ifndef WORDS_BIGENDIAN

        /* rgba -> bgra for rgba32 on little endian cpus */

        if (si->zsize == 4 && z != 3) 

            chan_offset = 2 - z;

        else

#endif

            chan_offset = z;

            

        for (y = si->ysize - 1; y >= 0; y--) {

            dest_row = pict->data[0] + (y * si->xsize * si->zsize);



            if (!get_buffer(f, tmp_row, si->xsize)) {

                ret = -1;

                goto cleanup;

            }

            for (x = 0; x < si->xsize; x++) {

                dest_row[chan_offset] = tmp_row[x]; 

                dest_row += si->zsize;

            }

        }

    }



cleanup:

    av_free(tmp_row);

    return ret;

}
