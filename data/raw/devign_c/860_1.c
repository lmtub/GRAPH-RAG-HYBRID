static void iscsi_refresh_limits(BlockDriverState *bs, Error **errp)

{

    /* We don't actually refresh here, but just return data queried in

     * iscsi_open(): iscsi targets don't change their limits. */



    IscsiLun *iscsilun = bs->opaque;

    uint64_t max_xfer_len = iscsilun->use_16_for_rw ? 0xffffffff : 0xffff;



    bs->bl.request_alignment = iscsilun->block_size;



    if (iscsilun->bl.max_xfer_len) {

        max_xfer_len = MIN(max_xfer_len, iscsilun->bl.max_xfer_len);

    }



    if (max_xfer_len * iscsilun->block_size < INT_MAX) {

        bs->bl.max_transfer = max_xfer_len * iscsilun->block_size;

    }



    if (iscsilun->lbp.lbpu) {

        if (iscsilun->bl.max_unmap < 0xffffffff / iscsilun->block_size) {

            bs->bl.max_pdiscard =

                iscsilun->bl.max_unmap * iscsilun->block_size;

        }

        bs->bl.pdiscard_alignment =

            iscsilun->bl.opt_unmap_gran * iscsilun->block_size;

    } else {

        bs->bl.pdiscard_alignment = iscsilun->block_size;

    }



    if (iscsilun->bl.max_ws_len < 0xffffffff / iscsilun->block_size) {

        bs->bl.max_pwrite_zeroes =

            iscsilun->bl.max_ws_len * iscsilun->block_size;

    }

    if (iscsilun->lbp.lbpws) {

        bs->bl.pwrite_zeroes_alignment =

            iscsilun->bl.opt_unmap_gran * iscsilun->block_size;

    } else {

        bs->bl.pwrite_zeroes_alignment = iscsilun->block_size;

    }

    if (iscsilun->bl.opt_xfer_len &&

        iscsilun->bl.opt_xfer_len < INT_MAX / iscsilun->block_size) {

        bs->bl.opt_transfer = pow2floor(iscsilun->bl.opt_xfer_len *

                                        iscsilun->block_size);

    }

}
