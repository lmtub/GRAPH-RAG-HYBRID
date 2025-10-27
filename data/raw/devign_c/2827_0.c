static void virtio_scsi_do_tmf(VirtIOSCSI *s, VirtIOSCSIReq *req)

{

    SCSIDevice *d = virtio_scsi_device_find(s, req->req.tmf->lun);

    SCSIRequest *r, *next;

    BusChild *kid;

    int target;



    /* Here VIRTIO_SCSI_S_OK means "FUNCTION COMPLETE".  */

    req->resp.tmf->response = VIRTIO_SCSI_S_OK;



    tswap32s(&req->req.tmf->subtype);

    switch (req->req.tmf->subtype) {

    case VIRTIO_SCSI_T_TMF_ABORT_TASK:

    case VIRTIO_SCSI_T_TMF_QUERY_TASK:

        if (!d) {

            goto fail;

        }

        if (d->lun != virtio_scsi_get_lun(req->req.tmf->lun)) {

            goto incorrect_lun;

        }

        QTAILQ_FOREACH_SAFE(r, &d->requests, next, next) {

            VirtIOSCSIReq *cmd_req = r->hba_private;

            if (cmd_req && cmd_req->req.cmd->tag == req->req.tmf->tag) {

                break;

            }

        }

        if (r) {

            /*

             * Assert that the request has not been completed yet, we

             * check for it in the loop above.

             */

            assert(r->hba_private);

            if (req->req.tmf->subtype == VIRTIO_SCSI_T_TMF_QUERY_TASK) {

                /* "If the specified command is present in the task set, then

                 * return a service response set to FUNCTION SUCCEEDED".

                 */

                req->resp.tmf->response = VIRTIO_SCSI_S_FUNCTION_SUCCEEDED;

            } else {

                scsi_req_cancel(r);

            }

        }

        break;



    case VIRTIO_SCSI_T_TMF_LOGICAL_UNIT_RESET:

        if (!d) {

            goto fail;

        }

        if (d->lun != virtio_scsi_get_lun(req->req.tmf->lun)) {

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

        if (d->lun != virtio_scsi_get_lun(req->req.tmf->lun)) {

            goto incorrect_lun;

        }

        QTAILQ_FOREACH_SAFE(r, &d->requests, next, next) {

            if (r->hba_private) {

                if (req->req.tmf->subtype == VIRTIO_SCSI_T_TMF_QUERY_TASK_SET) {

                    /* "If there is any command present in the task set, then

                     * return a service response set to FUNCTION SUCCEEDED".

                     */

                    req->resp.tmf->response = VIRTIO_SCSI_S_FUNCTION_SUCCEEDED;

                    break;

                } else {

                    scsi_req_cancel(r);

                }

            }

        }

        break;



    case VIRTIO_SCSI_T_TMF_I_T_NEXUS_RESET:

        target = req->req.tmf->lun[1];

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

        req->resp.tmf->response = VIRTIO_SCSI_S_FUNCTION_REJECTED;

        break;

    }



    return;



incorrect_lun:

    req->resp.tmf->response = VIRTIO_SCSI_S_INCORRECT_LUN;

    return;



fail:

    req->resp.tmf->response = VIRTIO_SCSI_S_BAD_TARGET;

}
