static int vscsi_srp_direct_data(VSCSIState *s, vscsi_req *req,

                                 uint8_t *buf, uint32_t len)

{

    struct srp_direct_buf *md = req->cur_desc;

    uint32_t llen;

    int rc;



    dprintf("VSCSI: direct segment 0x%x bytes, va=0x%llx desc len=0x%x\n",

            len, (unsigned long long)md->va, md->len);



    llen = MIN(len, md->len);

    if (llen) {

        if (req->writing) { /* writing = to device = reading from memory */

            rc = spapr_tce_dma_read(&s->vdev, md->va, buf, llen);

        } else {

            rc = spapr_tce_dma_write(&s->vdev, md->va, buf, llen);

        }

    }

    md->len -= llen;

    md->va += llen;



    if (rc) {

        return -1;

    }

    return llen;

}
