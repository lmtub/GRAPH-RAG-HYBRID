void nbd_client_session_attach_aio_context(NbdClientSession *client,

                                           AioContext *new_context)

{

    aio_set_fd_handler(new_context, client->sock,

                       nbd_reply_ready, NULL, client);

}
