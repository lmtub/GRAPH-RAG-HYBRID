static int virtio_scsi_do_tmf(VirtIOSCSI *s, VirtIOSCSIReq *req)

{

    SCSIDevice *d = virtio_scsi_device_find(s, req->req.tmf.lun);

    SCSIRequest *r, *next;

    BusChild *kid;

    int target;

    int ret = 0;



    if (s->dataplane_started) {

        assert(blk_get_aio_context(d->conf.blk) == s->ctx);

    }

    /* Here VIRTIO_SCSI_S_OK means "FUNCTION COMPLETE".  */

    req->resp.tmf.response = VIRTIO_SCSI_S_OK;



    virtio_tswap32s(VIRTIO_DEVICE(s), &req->req.tmf.subtype);

    switch (req->req.tmf.subtype) {

    case VIRTIO_SCSI_T_TMF_ABORT_TASK:

    case VIRTIO_SCSI_T_TMF_QUERY_TASK:

        if (!d) {

            goto fail;

        }

        if (d->lun != virtio_scsi_get_lun(req->req.tmf.lun)) {

            goto incorrect_lun;

        }

        QTAILQ_FOREACH_SAFE(r, &d->requests, next, next) {

            VirtIOSCSIReq *cmd_req = r->hba_private;

            if (cmd_req && cmd_req->req.cmd.tag == req->req.tmf.tag) {

                break;

            }

        }

        if (r) {

            /*

             * Assert that the request has not been completed yet, we

             * check for it in the loop above.

             */

            assert(r->hba_private);

            if (req->req.tmf.subtype == VIRTIO_SCSI_T_TMF_QUERY_TASK) {

                /* "If the specified command is present in the task set, then

                 * return a service response set to FUNCTION SUCCEEDED".

                 */

                req->resp.tmf.response = VIRTIO_SCSI_S_FUNCTION_SUCCEEDED;

            } else {

                VirtIOSCSICancelNotifier *notifier;



                req->remaining = 1;

                notifier = g_new(VirtIOSCSICancelNotifier, 1);

                notifier->tmf_req = req;

                notifier->notifier.notify = virtio_scsi_cancel_notify;

                scsi_req_cancel_async(r, &notifier->notifier);

                ret = -EINPROGRESS;

            }

        }

        break;



    case VIRTIO_SCSI_T_TMF_LOGICAL_UNIT_RESET:

        if (!d) {

            goto fail;

        }

        if (d->lun != virtio_scsi_get_lun(req->req.tmf.lun)) {

            goto incorrect_lun;

        }

        s->resetting++;

        qdev_reset_all(&d->qdev);

        s->resetting--;

        break;



    case VIRTIO_SCSI_T_TMF_ABORT_TASK_SET:

    case VIRTIO_SCSI_T_TMF_CLEAR_TASK_SET:

    case VIRTIO_SCSI_T_TMF_QUERY_TASK_SET:

        if (!d) {

            goto fail;

        }

        if (d->lun != virtio_scsi_get_lun(req->req.tmf.lun)) {

            goto incorrect_lun;

        }



        /* Add 1 to "remaining" until virtio_scsi_do_tmf returns.

         * This way, if the bus starts calling back to the notifiers

         * even before we finish the loop, virtio_scsi_cancel_notify

         * will not complete the TMF too early.

         */

        req->remaining = 1;

        QTAILQ_FOREACH_SAFE(r, &d->requests, next, next) {

            if (r->hba_private) {

                if (req->req.tmf.subtype == VIRTIO_SCSI_T_TMF_QUERY_TASK_SET) {

                    /* "If there is any command present in the task set, then

                     * return a service response set to FUNCTION SUCCEEDED".

                     */

                    req->resp.tmf.response = VIRTIO_SCSI_S_FUNCTION_SUCCEEDED;

                    break;

                } else {

                    VirtIOSCSICancelNotifier *notifier;



                    req->remaining++;

                    notifier = g_new(VirtIOSCSICancelNotifier, 1);

                    notifier->notifier.notify = virtio_scsi_cancel_notify;

                    notifier->tmf_req = req;

                    scsi_req_cancel_async(r, &notifier->notifier);

                }

            }

        }

        if (--req->remaining > 0) {

            ret = -EINPROGRESS;

        }

        break;



    case VIRTIO_SCSI_T_TMF_I_T_NEXUS_RESET:

        target = req->req.tmf.lun[1];

        s->resetting++;

        QTAILQ_FOREACH(kid, &s->bus.qbus.children, sibling) {

             d = DO_UPCAST(SCSIDevice, qdev, kid->child);

             if (d->channel == 0 && d->id == target) {

                qdev_reset_all(&d->qdev);

             }

        }

        s->resetting--;

        break;



    case VIRTIO_SCSI_T_TMF_CLEAR_ACA:

    default:

        req->resp.tmf.response = VIRTIO_SCSI_S_FUNCTION_REJECTED;

        break;

    }



    return ret;



incorrect_lun:

    req->resp.tmf.response = VIRTIO_SCSI_S_INCORRECT_LUN;

    return ret;



fail:

    req->resp.tmf.response = VIRTIO_SCSI_S_BAD_TARGET;

    return ret;

}
