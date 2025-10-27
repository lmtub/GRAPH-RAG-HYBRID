static int tight_init_stream(VncState *vs, int stream_id,

                             int level, int strategy)

{

    z_streamp zstream = &vs->tight_stream[stream_id];



    if (zstream->opaque == NULL) {

        int err;



        VNC_DEBUG("VNC: TIGHT: initializing zlib stream %d\n", stream_id);

        VNC_DEBUG("VNC: TIGHT: opaque = %p | vs = %p\n", zstream->opaque, vs);

        zstream->zalloc = vnc_zlib_zalloc;

        zstream->zfree = vnc_zlib_zfree;



        err = deflateInit2(zstream, level, Z_DEFLATED, MAX_WBITS,

                           MAX_MEM_LEVEL, strategy);



        if (err != Z_OK) {

            fprintf(stderr, "VNC: error initializing zlib\n");

            return -1;

        }



        vs->tight_levels[stream_id] = level;

        zstream->opaque = vs;

    }



    if (vs->tight_levels[stream_id] != level) {

        if (deflateParams(zstream, level, strategy) != Z_OK) {

            return -1;

        }

        vs->tight_levels[stream_id] = level;

    }

    return 0;

}
