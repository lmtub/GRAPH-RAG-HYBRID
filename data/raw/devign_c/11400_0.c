send_msg(

    VSCMsgType type,

    uint32_t reader_id,

    const void *msg,

    unsigned int length

) {

    VSCMsgHeader mhHeader;



    qemu_mutex_lock(&socket_to_send_lock);



    if (verbose > 10) {

        printf("sending type=%d id=%u, len =%u (0x%x)\n",

               type, reader_id, length, length);

    }



    mhHeader.type = htonl(type);

    mhHeader.reader_id = 0;

    mhHeader.length = htonl(length);

    g_byte_array_append(socket_to_send, (guint8 *)&mhHeader, sizeof(mhHeader));

    g_byte_array_append(socket_to_send, (guint8 *)msg, length);

    g_idle_add(socket_prepare_sending, NULL);



    qemu_mutex_unlock(&socket_to_send_lock);



    return 0;

}
