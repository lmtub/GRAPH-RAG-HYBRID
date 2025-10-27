static void h264_free_context(PayloadContext *data)

{

#ifdef DEBUG

    int ii;



    for (ii = 0; ii < 32; ii++) {

        if (data->packet_types_received[ii])

            av_log(NULL, AV_LOG_DEBUG, "Received %d packets of type %d\n",

                   data->packet_types_received[ii], ii);

    }

#endif



    assert(data);

    assert(data->cookie == MAGIC_COOKIE);



    // avoid stale pointers (assert)

    data->cookie = DEAD_COOKIE;



    // and clear out this...

    av_free(data);

}
