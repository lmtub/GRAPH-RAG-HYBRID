static void virtio_blk_handle_scsi(VirtIOBlockReq *req)

{

    int status;



    status = virtio_blk_handle_scsi_req(req->dev, req->elem);

    virtio_blk_req_complete(req, status);

    virtio_blk_free_request(req);

}
