static int read_write_object(int fd, char *buf, uint64_t oid, int copies,

                             unsigned int datalen, uint64_t offset,

                             bool write, bool create, bool cache)

{

    SheepdogObjReq hdr;

    SheepdogObjRsp *rsp = (SheepdogObjRsp *)&hdr;

    unsigned int wlen, rlen;

    int ret;



    memset(&hdr, 0, sizeof(hdr));



    if (write) {

        wlen = datalen;

        rlen = 0;

        hdr.flags = SD_FLAG_CMD_WRITE;

        if (create) {

            hdr.opcode = SD_OP_CREATE_AND_WRITE_OBJ;

        } else {

            hdr.opcode = SD_OP_WRITE_OBJ;

        }

    } else {

        wlen = 0;

        rlen = datalen;

        hdr.opcode = SD_OP_READ_OBJ;

    }



    if (cache) {

        hdr.flags |= SD_FLAG_CMD_CACHE;

    }



    hdr.oid = oid;

    hdr.data_length = datalen;

    hdr.offset = offset;

    hdr.copies = copies;



    ret = do_req(fd, (SheepdogReq *)&hdr, buf, &wlen, &rlen);

    if (ret) {

        error_report("failed to send a request to the sheep");

        return ret;

    }



    switch (rsp->result) {

    case SD_RES_SUCCESS:

        return 0;

    default:

        error_report("%s", sd_strerror(rsp->result));

        return -EIO;

    }

}
