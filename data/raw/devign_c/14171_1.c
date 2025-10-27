static int coroutine_fn sd_co_rw_vector(void *p)

{

    SheepdogAIOCB *acb = p;

    int ret = 0;

    unsigned long len, done = 0, total = acb->nb_sectors * BDRV_SECTOR_SIZE;

    unsigned long idx = acb->sector_num * BDRV_SECTOR_SIZE / SD_DATA_OBJ_SIZE;

    uint64_t oid;

    uint64_t offset = (acb->sector_num * BDRV_SECTOR_SIZE) % SD_DATA_OBJ_SIZE;

    BDRVSheepdogState *s = acb->common.bs->opaque;

    SheepdogInode *inode = &s->inode;

    AIOReq *aio_req;



    if (acb->aiocb_type == AIOCB_WRITE_UDATA && s->is_snapshot) {

        /*

         * In the case we open the snapshot VDI, Sheepdog creates the

         * writable VDI when we do a write operation first.

         */

        ret = sd_create_branch(s);

        if (ret) {

            acb->ret = -EIO;

            goto out;

        }

    }



    /*

     * Make sure we don't free the aiocb before we are done with all requests.

     * This additional reference is dropped at the end of this function.

     */

    acb->nr_pending++;



    while (done != total) {

        uint8_t flags = 0;

        uint64_t old_oid = 0;

        bool create = false;



        oid = vid_to_data_oid(inode->data_vdi_id[idx], idx);



        len = MIN(total - done, SD_DATA_OBJ_SIZE - offset);



        switch (acb->aiocb_type) {

        case AIOCB_READ_UDATA:

            if (!inode->data_vdi_id[idx]) {

                qemu_iovec_memset(acb->qiov, done, 0, len);

                goto done;

            }

            break;

        case AIOCB_WRITE_UDATA:

            if (!inode->data_vdi_id[idx]) {

                create = true;

            } else if (!is_data_obj_writable(inode, idx)) {

                /* Copy-On-Write */

                create = true;

                old_oid = oid;

                flags = SD_FLAG_CMD_COW;

            }

            break;

        case AIOCB_DISCARD_OBJ:

            /*

             * We discard the object only when the whole object is

             * 1) allocated 2) trimmed. Otherwise, simply skip it.

             */

            if (len != SD_DATA_OBJ_SIZE || inode->data_vdi_id[idx] == 0) {

                goto done;

            }

            break;

        default:

            break;

        }



        if (create) {

            DPRINTF("update ino (%" PRIu32 ") %" PRIu64 " %" PRIu64 " %ld\n",

                    inode->vdi_id, oid,

                    vid_to_data_oid(inode->data_vdi_id[idx], idx), idx);

            oid = vid_to_data_oid(inode->vdi_id, idx);

            DPRINTF("new oid %" PRIx64 "\n", oid);

        }



        aio_req = alloc_aio_req(s, acb, oid, len, offset, flags, old_oid, done);



        if (create) {

            AIOReq *areq;

            QLIST_FOREACH(areq, &s->inflight_aio_head, aio_siblings) {

                if (areq->oid == oid) {

                    /*

                     * Sheepdog cannot handle simultaneous create

                     * requests to the same object.  So we cannot send

                     * the request until the previous request

                     * finishes.

                     */

                    aio_req->flags = 0;

                    aio_req->base_oid = 0;

                    QLIST_INSERT_HEAD(&s->pending_aio_head, aio_req,

                                      aio_siblings);

                    goto done;

                }

            }

        }



        QLIST_INSERT_HEAD(&s->inflight_aio_head, aio_req, aio_siblings);

        add_aio_request(s, aio_req, acb->qiov->iov, acb->qiov->niov, create,

                        acb->aiocb_type);

    done:

        offset = 0;

        idx++;

        done += len;

    }

out:

    if (!--acb->nr_pending) {

        return acb->ret;

    }

    return 1;

}
