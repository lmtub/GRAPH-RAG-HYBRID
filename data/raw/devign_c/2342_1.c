int nbd_client_co_pwrite_zeroes(BlockDriverState *bs, int64_t offset,

                                int bytes, BdrvRequestFlags flags)

{

    NBDClientSession *client = nbd_get_client_session(bs);

    NBDRequest request = {

        .type = NBD_CMD_WRITE_ZEROES,

        .from = offset,

        .len = bytes,

    };




    if (!(client->info.flags & NBD_FLAG_SEND_WRITE_ZEROES)) {

        return -ENOTSUP;

    }



    if (flags & BDRV_REQ_FUA) {

        assert(client->info.flags & NBD_FLAG_SEND_FUA);

        request.flags |= NBD_CMD_FLAG_FUA;

    }

    if (!(flags & BDRV_REQ_MAY_UNMAP)) {

        request.flags |= NBD_CMD_FLAG_NO_HOLE;

    }



    return nbd_co_request(bs, &request, NULL);

}