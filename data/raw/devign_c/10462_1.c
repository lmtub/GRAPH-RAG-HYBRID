int nbd_client_co_pdiscard(BlockDriverState *bs, int64_t offset, int bytes)

{

    NBDClientSession *client = nbd_get_client_session(bs);

    NBDRequest request = {

        .type = NBD_CMD_TRIM,

        .from = offset,

        .len = bytes,

    };




    if (!(client->info.flags & NBD_FLAG_SEND_TRIM)) {

        return 0;

    }



    return nbd_co_request(bs, &request, NULL);

}