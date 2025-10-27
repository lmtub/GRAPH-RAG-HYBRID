static uint32_t scsi_init_iovec(SCSIDiskReq *r, size_t size)

{

    SCSIDiskState *s = DO_UPCAST(SCSIDiskState, qdev, r->req.dev);



    if (!r->iov.iov_base) {

        r->buflen = size;

        r->iov.iov_base = qemu_blockalign(s->qdev.conf.bs, r->buflen);

    }

    r->iov.iov_len = MIN(r->sector_count * 512, r->buflen);

    qemu_iovec_init_external(&r->qiov, &r->iov, 1);

    return r->qiov.size / 512;

}
