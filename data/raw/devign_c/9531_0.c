static int qed_write_header_sync(BDRVQEDState *s)

{

    QEDHeader le;

    int ret;



    qed_header_cpu_to_le(&s->header, &le);

    ret = bdrv_pwrite(s->bs->file, 0, &le, sizeof(le));

    if (ret != sizeof(le)) {

        return ret;

    }

    return 0;

}
