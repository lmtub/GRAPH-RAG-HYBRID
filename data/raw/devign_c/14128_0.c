static void sd_close(BlockDriverState *bs)

{

    Error *local_err = NULL;

    BDRVSheepdogState *s = bs->opaque;

    SheepdogVdiReq hdr;

    SheepdogVdiRsp *rsp = (SheepdogVdiRsp *)&hdr;

    unsigned int wlen, rlen = 0;

    int fd, ret;



    DPRINTF("%s\n", s->name);



    fd = connect_to_sdog(s, &local_err);

    if (fd < 0) {

        error_report_err(local_err);

        return;

    }



    memset(&hdr, 0, sizeof(hdr));



    hdr.opcode = SD_OP_RELEASE_VDI;

    hdr.type = LOCK_TYPE_NORMAL;

    hdr.base_vdi_id = s->inode.vdi_id;

    wlen = strlen(s->name) + 1;

    hdr.data_length = wlen;

    hdr.flags = SD_FLAG_CMD_WRITE;



    ret = do_req(fd, s->bs, (SheepdogReq *)&hdr,

                 s->name, &wlen, &rlen);



    closesocket(fd);



    if (!ret && rsp->result != SD_RES_SUCCESS &&

        rsp->result != SD_RES_VDI_NOT_LOCKED) {

        error_report("%s, %s", sd_strerror(rsp->result), s->name);

    }



    aio_set_fd_handler(bdrv_get_aio_context(bs), s->fd,

                       false, NULL, NULL, NULL, NULL);

    closesocket(s->fd);

    qapi_free_SocketAddressLegacy(s->addr);

}
