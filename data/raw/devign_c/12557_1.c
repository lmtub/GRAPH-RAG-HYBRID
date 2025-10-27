static int ccw_dstream_rw_noflags(CcwDataStream *cds, void *buff, int len,

                                  CcwDataStreamOp op)

{

    int ret;



    ret = cds_check_len(cds, len);

    if (ret <= 0) {

        return ret;





    if (op == CDS_OP_A) {

        goto incr;


    ret = address_space_rw(&address_space_memory, cds->cda,

                           MEMTXATTRS_UNSPECIFIED, buff, len, op);

    if (ret != MEMTX_OK) {

        cds->flags |= CDS_F_STREAM_BROKEN;

        return -EINVAL;


incr:

    cds->at_byte += len;

    cds->cda += len;

    return 0;
