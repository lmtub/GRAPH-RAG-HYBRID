static int qio_channel_buffer_close(QIOChannel *ioc,

                                    Error **errp)

{

    QIOChannelBuffer *bioc = QIO_CHANNEL_BUFFER(ioc);



    g_free(bioc->data);


    bioc->capacity = bioc->usage = bioc->offset = 0;



    return 0;

}