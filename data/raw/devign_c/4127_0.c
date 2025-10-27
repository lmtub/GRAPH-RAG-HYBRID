static void coroutine_fn aio_read_response(void *opaque)

{

    SheepdogObjRsp rsp;

    BDRVSheepdogState *s = opaque;

    int fd = s->fd;

    int ret;

    AIOReq *aio_req = NULL;

    SheepdogAIOCB *acb;

    uint64_t idx;



    if (QLIST_EMPTY(&s->inflight_aio_head)) {

        goto out;

    }



    /* read a header */

    ret = qemu_co_recv(fd, &rsp, sizeof(rsp));

    if (ret < 0) {

        error_report("failed to get the header, %s", strerror(errno));

        goto out;

    }



    /* find the right aio_req from the inflight aio list */

    QLIST_FOREACH(aio_req, &s->inflight_aio_head, aio_siblings) {

        if (aio_req->id == rsp.id) {

            break;

        }

    }

    if (!aio_req) {

        error_report("cannot find aio_req %x", rsp.id);

        goto out;

    }



    acb = aio_req->aiocb;



    switch (acb->aiocb_type) {

    case AIOCB_WRITE_UDATA:

        /* this coroutine context is no longer suitable for co_recv

         * because we may send data to update vdi objects */

        s->co_recv = NULL;

        if (!is_data_obj(aio_req->oid)) {

            break;

        }

        idx = data_oid_to_idx(aio_req->oid);



        if (s->inode.data_vdi_id[idx] != s->inode.vdi_id) {

            /*

             * If the object is newly created one, we need to update

             * the vdi object (metadata object).  min_dirty_data_idx

             * and max_dirty_data_idx are changed to include updated

             * index between them.

             */

            if (rsp.result == SD_RES_SUCCESS) {

                s->inode.data_vdi_id[idx] = s->inode.vdi_id;

                s->max_dirty_data_idx = MAX(idx, s->max_dirty_data_idx);

                s->min_dirty_data_idx = MIN(idx, s->min_dirty_data_idx);

            }

            /*

             * Some requests may be blocked because simultaneous

             * create requests are not allowed, so we search the

             * pending requests here.

             */

            send_pending_req(s, aio_req->oid);

        }

        break;

    case AIOCB_READ_UDATA:

        ret = qemu_co_recvv(fd, acb->qiov->iov, acb->qiov->niov,

                            aio_req->iov_offset, rsp.data_length);

        if (ret < 0) {

            error_report("failed to get the data, %s", strerror(errno));

            goto out;

        }

        break;

    case AIOCB_FLUSH_CACHE:

        if (rsp.result == SD_RES_INVALID_PARMS) {

            DPRINTF("disable cache since the server doesn't support it\n");

            s->cache_flags = SD_FLAG_CMD_DIRECT;

            rsp.result = SD_RES_SUCCESS;

        }

        break;

    case AIOCB_DISCARD_OBJ:

        switch (rsp.result) {

        case SD_RES_INVALID_PARMS:

            error_report("sheep(%s) doesn't support discard command",

                         s->host_spec);

            rsp.result = SD_RES_SUCCESS;

            s->discard_supported = false;

            break;

        case SD_RES_SUCCESS:

            idx = data_oid_to_idx(aio_req->oid);

            s->inode.data_vdi_id[idx] = 0;

            break;

        default:

            break;

        }

    }



    switch (rsp.result) {

    case SD_RES_SUCCESS:

        break;

    case SD_RES_READONLY:

        ret = resend_aioreq(s, aio_req);

        if (ret == SD_RES_SUCCESS) {

            goto out;

        }

        /* fall through */

    default:

        acb->ret = -EIO;

        error_report("%s", sd_strerror(rsp.result));

        break;

    }



    free_aio_req(s, aio_req);

    if (!acb->nr_pending) {

        /*

         * We've finished all requests which belong to the AIOCB, so

         * we can switch back to sd_co_readv/writev now.

         */

        acb->aio_done_func(acb);

    }

out:

    s->co_recv = NULL;

}
