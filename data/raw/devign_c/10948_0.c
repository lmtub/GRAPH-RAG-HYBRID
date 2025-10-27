int nbd_trip(BlockDriverState *bs, int csock, off_t size, uint64_t dev_offset,

             off_t *offset, uint32_t nbdflags, uint8_t *data, int data_size)

{

    struct nbd_request request;

    struct nbd_reply reply;



    TRACE("Reading request.");



    if (nbd_receive_request(csock, &request) == -1)

        return -1;



    if (request.len + NBD_REPLY_SIZE > data_size) {

        LOG("len (%u) is larger than max len (%u)",

            request.len + NBD_REPLY_SIZE, data_size);

        errno = EINVAL;

        return -1;

    }



    if ((request.from + request.len) < request.from) {

        LOG("integer overflow detected! "

            "you're probably being attacked");

        errno = EINVAL;

        return -1;

    }



    if ((request.from + request.len) > size) {

            LOG("From: %" PRIu64 ", Len: %u, Size: %" PRIu64

            ", Offset: %" PRIu64 "\n",

                    request.from, request.len, (uint64_t)size, dev_offset);

        LOG("requested operation past EOF--bad client?");

        errno = EINVAL;

        return -1;

    }



    TRACE("Decoding type");



    reply.handle = request.handle;

    reply.error = 0;



    switch (request.type) {

    case NBD_CMD_READ:

        TRACE("Request type is READ");



        if (bdrv_read(bs, (request.from + dev_offset) / 512,

                  data + NBD_REPLY_SIZE,

                  request.len / 512) == -1) {

            LOG("reading from file failed");

            errno = EINVAL;

            return -1;

        }

        *offset += request.len;



        TRACE("Read %u byte(s)", request.len);



        /* Reply

           [ 0 ..  3]    magic   (NBD_REPLY_MAGIC)

           [ 4 ..  7]    error   (0 == no error)

           [ 7 .. 15]    handle

         */



        cpu_to_be32w((uint32_t*)data, NBD_REPLY_MAGIC);

        cpu_to_be32w((uint32_t*)(data + 4), reply.error);

        cpu_to_be64w((uint64_t*)(data + 8), reply.handle);



        TRACE("Sending data to client");



        if (write_sync(csock, data,

                   request.len + NBD_REPLY_SIZE) !=

                   request.len + NBD_REPLY_SIZE) {

            LOG("writing to socket failed");

            errno = EINVAL;

            return -1;

        }

        break;

    case NBD_CMD_WRITE:

        TRACE("Request type is WRITE");



        TRACE("Reading %u byte(s)", request.len);



        if (read_sync(csock, data, request.len) != request.len) {

            LOG("reading from socket failed");

            errno = EINVAL;

            return -1;

        }



        if (nbdflags & NBD_FLAG_READ_ONLY) {

            TRACE("Server is read-only, return error");

            reply.error = 1;

        } else {

            TRACE("Writing to device");



            if (bdrv_write(bs, (request.from + dev_offset) / 512,

                       data, request.len / 512) == -1) {

                LOG("writing to file failed");

                errno = EINVAL;

                return -1;

            }



            *offset += request.len;

        }



        if (nbd_send_reply(csock, &reply) == -1)

            return -1;

        break;

    case NBD_CMD_DISC:

        TRACE("Request type is DISCONNECT");

        errno = 0;

        return 1;

    default:

        LOG("invalid request type (%u) received", request.type);

        errno = EINVAL;

        return -1;

    }



    TRACE("Request/Reply complete");



    return 0;

}
